import torch
import torch.nn as nn
import os
import time

import torch.optim as optim
from torch.autograd import Variable
import src.utilities.utils as utils
import numpy as np
import cv2
from copy import deepcopy

import src.methods.DDGR.train_DDGR as DDGR_train
from src.data.imgfolder import ImageFolderTrainVal

import torch.distributed as dist
import torch.nn.functional as F

from src.methods.DDGR.diffusion import dist_util, logger
from src.methods.DDGR.diffusion.image_datasets import load_data
from src.methods.DDGR.diffusion.resample import create_named_schedule_sampler
from src.methods.DDGR.diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    args_to_dict,
)
from src.methods.DDGR.diffusion.train_util import TrainLoop


def fine_tune_train_DDGR(dataset_path, args,previous_task_model_path, exp_dir, task_counter, batch_size=200,
                               num_epochs=100, lr=0.0008, weight_decay=0, head_shared=False,
                               saving_freq=5,classifier_type="pretrained"):

    start_mem = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    if not os.path.exists(exp_dir):
        print("Going to exp_dir=", exp_dir)
        os.makedirs(exp_dir)

    dist_util.setup_dist()
    utils.create_dir(os.path.join(exp_dir,"logger_file"))
    logger.configure(dir=os.path.join(exp_dir,"logger_file"))
    sample_start_time=None
    sample_end_time=None
    since = time.time()
    use_cuda = torch.cuda.is_available()
    dsets = torch.load(dataset_path)
    print(dataset_path)

    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    generator_img_path_label_list = None
    generator_classes = None
    generator_class_to_idx = None
    if task_counter > 0:
        print('load model')
        model_ft = torch.load(previous_task_model_path)
        if classifier_type=='self':
            dclassifier = deepcopy(model_ft)
        if not head_shared:
            last_layer_index = str(len(model_ft.classifier._modules) - 1)
            num_ftrs = model_ft.classifier._modules[last_layer_index].in_features
            model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))
            print("NEW FC CLASSIFIER HEAD with {} units".format(len(dset_classes)))
        criterion = nn.CrossEntropyLoss()
        temp_tensor = torch.zeros((2, 2)).cpu()
        if use_cuda:
            criterion.cuda()
            model_ft.cuda()
            temp_tensor = Variable(temp_tensor.cuda())
        device = temp_tensor.device
        tg_params = model_ft.parameters()
        optimizer_ft = optim.SGD(tg_params, lr=lr, momentum=0.9, weight_decay=weight_decay)

        print("Loard model and diffusion... args.num_classes={}".format(args.num_classes))
        dmodel, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        previous_diffusion_model_path = os.path.join(os.path.dirname(previous_task_model_path), 'model_000000.pt')
        dmodel.load_state_dict(
            dist_util.load_state_dict(previous_diffusion_model_path, map_location="cpu")
        )
        dmodel.to(dist_util.dev())
        samples_path = os.path.join(exp_dir,'samples')

        order_root_path = os.path.dirname(previous_task_model_path)
        order_name = os.path.join(order_root_path,
                                  "generator_label_list_order_seed={}.pkl".format(args.CI_order_rndseed))
        print("Order name:{}".format(order_name))
        label_order_list = utils.unpickle(order_name)

        pre_task_class_to_idx = {label_order_list[i]: i for i in range(len(label_order_list))}

        if not os.path.exists(samples_path):
            print('Sample imgs ...')

            if args.use_fp16:
                dmodel.convert_to_fp16()
            dmodel.eval()

            print("loading classifier...")
            if classifier_type=='self':
                dclassifier.to(dist_util.dev())
                dclassifier.eval()

                def cond_fn(x, t, y=None):
                    assert y is not None
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = dclassifier(x_in, label_order_list)
                        log_probs = F.log_softmax(logits, dim=-1)

                        l = [pre_task_class_to_idx[y[i].item()] for i in range(len(y))]
                        ll = torch.tensor(l).reshape(y.shape)
                        labels = ll.to(y.device)

                        selected = log_probs[range(len(logits)), labels.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

                def model_fn(x, t, y=None):
                    assert y is not None
                    return dmodel(x, t, y if args.class_cond else None)


                all_class_name = os.path.join(os.path.dirname(os.path.dirname(dataset_path)),"order_seed={}.pkl".format(args.CI_order_rndseed))
                all_class_name_list = utils.unpickle(all_class_name)
                print("sampling...")
                all_images = []
                all_labels = []
                if args.num_samples is not None:
                    num_samples = args.num_samples * len(label_order_list)
                    classifier_batch_size = args.num_samples
                else:
                    num_samples = dset_sizes['train']+dset_sizes['val']
                    num_samples = int(num_samples*args.DDGR_generator_factor)
                    num_samples = num_samples - num_samples % args.classifier_batch_size
                    classifier_batch_size = args.classifier_batch_size
                label_counter = 0
                sample_start_time = time.time()
                while len(all_images) * classifier_batch_size < num_samples:
                    model_kwargs = {}
                    if args.num_samples is not None:
                        rnd_label = np.array([label_order_list[label_counter] for _ in range(classifier_batch_size)])
                        label_counter += 1
                    else:
                        rnd_label = np.random.choice(label_order_list,size=(classifier_batch_size,))
                    classes = torch.from_numpy(rnd_label).to(dist_util.dev()).to(int)
                    model_kwargs["y"] = classes
                    sample_fn = (
                        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                    )
                    sample = sample_fn(
                        model_fn,
                        (classifier_batch_size, 3, args.image_size, args.image_size),
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                        cond_fn=cond_fn,
                        device=dist_util.dev(),
                    )
                    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                    sample = sample.permute(0, 2, 3, 1)
                    sample = sample.contiguous()

                    gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_samples, sample)
                    all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                    gathered_labels = [torch.zeros_like(classes) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_labels, classes)
                    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                    print(f"created {len(all_images) * classifier_batch_size} samples")
                utils.create_dir(samples_path)
                generator_img_path_label_list,generator_classes,generator_class_to_idx = save_to_JPEG(num_samples,
                                                                                                      all_images,
                                                                                                      all_labels,
                                                                                                      all_class_name_list,
                                                                                                      samples_path)
                sample_end_time = time.time()
            utils.savepickle(data=generator_img_path_label_list,file_path=os.path.join(exp_dir,"generator_img_path_label_list.pkl"))
            utils.savepickle(data=generator_classes,file_path=os.path.join(exp_dir, "generator_classes.pkl"))
            utils.savepickle(data=generator_class_to_idx, file_path=os.path.join(exp_dir, "generator_class_to_idx.pkl"))

        else:
            print(print('Samples are in {}'.format(samples_path)))
            generator_img_path_label_list = utils.unpickle(os.path.join(exp_dir,"generator_img_path_label_list.pkl"))
            generator_classes = utils.unpickle(os.path.join(exp_dir,"generator_classes.pkl"))
            generator_class_to_idx = utils.unpickle(os.path.join(exp_dir,"generator_class_to_idx.pkl"))

    else:
        print("Loading prev model from path: ", previous_task_model_path)
        model_ft = torch.load(previous_task_model_path)
        if not head_shared:
            last_layer_index = str(len(model_ft.classifier._modules) - 1)
            num_ftrs = model_ft.classifier._modules[last_layer_index].in_features
            model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))
            print("NEW FC CLASSIFIER HEAD with {} units".format(len(dset_classes)))
        criterion = nn.CrossEntropyLoss()
        temp_tensor = torch.zeros((2,2)).cpu()
        if use_cuda:
            criterion.cuda()
            model_ft.cuda()
            temp_tensor = Variable(temp_tensor.cuda())

        device = temp_tensor.device
        tg_params = model_ft.parameters()
        optimizer_ft = optim.SGD(tg_params, lr=lr, momentum=0.9, weight_decay=weight_decay)

        print("creating model and diffusion... args.num_classes={}".format(args.num_classes))
        dmodel, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        dmodel.to(dist_util.dev())
        samples_path = os.path.join(exp_dir, 'samples')
        label_order_list = []

    ddsets = None
    if (generator_img_path_label_list is not None) and (generator_classes is not None) and (generator_class_to_idx is not None):
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                       shuffle=True, pin_memory=True)
                        for x in ['train', 'val']}
        ddsets = dsets

        gen_dset = combine_data_loader(generator_img_path_label_list,
                                           generator_classes,
                                           generator_class_to_idx,
                                           dsets)
        gen_dset_loaders = {x: torch.utils.data.DataLoader(gen_dset[x], batch_size=batch_size,
                                                       shuffle=True, pin_memory=True)
                        for x in ['train', 'val']}
        gen_dset_sizes = {x: len(gen_dset[x]) for x in ['train', 'val']}
    else:
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                       shuffle=True, pin_memory=True)
                        for x in ['train', 'val']}
        ddsets = dsets
        gen_dset = None
        gen_dset_loaders = None
        gen_dset_sizes = None

    pl_list = ddsets['train'].get_allfigs_filepath()
    dset_path_list = [item[0] for item in pl_list]
    labels_list = [item[1] for item in pl_list]

    temp_ll = deepcopy(labels_list)
    temp_ll.extend(label_order_list)
    combine_label_list = list(sorted(set(temp_ll)))

    if (gen_dset is not None) and (gen_dset_loaders is not None) and (gen_dset_sizes is not None):
        gen_pl_list = gen_dset['train'].get_allfigs_filepath()
        gen_dset_path_list = [item[0] for item in gen_pl_list]
        gen_labels_list = [item[1] for item in gen_pl_list]
        dset_path_list.extend(gen_dset_path_list)
        labels_list.extend(gen_labels_list)

    resume = os.path.join(exp_dir, 'epoch.pth.tar')

    model_ft, best_acc = DDGR_train.train_model(args=args,model=model_ft, criterion=criterion,
                                                optimizer=optimizer_ft,lr = lr,
                                                dset_loaders = dset_loaders, dset_sizes = dset_sizes,
                                                use_cuda = use_cuda, num_epochs =num_epochs,task_counter = task_counter,
                                                exp_dir=exp_dir,resume=resume, saving_freq=saving_freq,device=device,
                                                combine_label_list=combine_label_list,gen_dset_loaders=gen_dset_loaders)



    print("DDGR training preparing")
    start_preprocess_time = time.time()

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    generator_data = load_data(data_dir=dset_path_list,
                               batch_size=args.diffusion_batch_size,
                               image_size=args.image_size,
                               classes_list=labels_list,
                               class_cond=args.class_cond,)
    print("DDGR training starts")
    TrainLoop(
        model=dmodel,
        diffusion=diffusion,
        data=generator_data,
        batch_size=args.diffusion_batch_size,
        microbatch=args.microbatch,
        lr=args.diffusion_lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.diffusion_weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        exp_dir=exp_dir
    ).run_loop()

    time_cost = time.time() - since
    if sample_start_time is None and sample_end_time is None:
        sample_time=0
    else:
        sample_time=sample_end_time-sample_start_time
    save_runningtime(str(time_cost)+","+str(time.time()-start_preprocess_time)+","+str(sample_time), filename=exp_dir + '/' + 'runningtime.txt')
    end_mem = torch.cuda.max_memory_allocated()
    save_runningtime(str(start_mem)+","+str(end_mem),filename=exp_dir + '/' + 'mem.txt')
    combine_order_name = os.path.join(exp_dir,
                                      "generator_label_list_order_seed={}.pkl".format(args.CI_order_rndseed))
    print("Order name:{}".format(combine_order_name))
    print("Current label_list: {}".format(combine_label_list))
    utils.savepickle(combine_label_list, combine_order_name)
    return model_ft, best_acc

def combine_data_loader(generator_img_path_label_list,generator_classes,generator_class_to_idx,dset):
    gimg = deepcopy(generator_img_path_label_list)
    np.random.seed(7)
    np.random.shuffle(gimg)
    np.random.seed(7)
    gimg_split = {
        'train': [item for item in gimg[:round(0.8*len(gimg))]],
        'val': [item for item in gimg[round(0.8*len(gimg)):-1]]
    }
    combine_dset = {}
    for x in ['train', 'val']:
        imgs = deepcopy(gimg_split[x])
        root = deepcopy(dset[x].get_root())
        classes = deepcopy(generator_classes)
        classes.sort()
        class_to_idx = deepcopy(generator_class_to_idx)
        class_to_idx.sort()
        transform = deepcopy(dset[x].get_trans())
        combine_dset[x] = ImageFolderTrainVal(root,None,transform=transform,classes=classes,class_to_idx=class_to_idx,
                                              imgs=imgs)
    return combine_dset


def save_runningtime(time_elapsed,filename='runningtime.txt'):
    File = open(filename,'w')
    File.write(time_elapsed)
    File.close()

def save_to_JPEG(num_samples,all_images,all_labels,all_class_name,output_path):
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: num_samples]
    class_to_idx = set()
    classes = set()
    img_path_label_list = []
    if dist.get_rank() == 0:
        num_fig = arr.shape[0]
        for fig_count in range(num_fig):
            class_name = all_class_name[label_arr[fig_count]]
            class_to_idx.add(label_arr[fig_count])
            classes.add(class_name)
            filepath = os.path.join(output_path,
                                    "{}_generator{}.JPEG".format(class_name, fig_count))
            img_path_label_list.append((filepath,label_arr[fig_count]))
            output_img(filepath, np.array(arr[fig_count]))
    dist.barrier()
    class_to_idx = list(class_to_idx)
    class_to_idx.sort()
    classes = list(classes)
    classes.sort()
    print("sampling complete")
    return img_path_label_list,classes,class_to_idx
def output_img(filepath,img):
    cv2.imwrite(filepath,img)
