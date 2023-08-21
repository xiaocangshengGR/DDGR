import os

import torch
import torch.nn as nn
import torch.optim as optim

import src.utilities.utils as utils

import src.methods.Finetune.train_SGD as SGD_Training
import time
from copy import deepcopy
def fine_tune_SGD(dset_dataloader, cumsum_dset_sizes, dset_classes, model_path, exp_dir, num_epochs=100, lr=0.0004,
                  freeze_mode=0, weight_decay=0, enable_resume=True, replace_last_classifier_layer=True,
                  save_models_mode=True, freq=5, method_name = None, args=None,dset_class_to_idx=None):
    since = time.time()
    resume = os.path.join(exp_dir, 'epoch.pth.tar') if enable_resume else ''
    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        model_ft = checkpoint['model']
        print("Resumed from model: ", resume)
    else:
        if not os.path.exists(exp_dir) and save_models_mode:
            os.makedirs(exp_dir)
        if not os.path.isfile(model_path):
            raise Exception("Model path non-existing: {}".format(model_path))
        else:
            model_ft = torch.load(model_path)
            print("Starting from model path: ", model_path)


    use_gpu = torch.cuda.is_available()

    criterion = nn.CrossEntropyLoss()

    if isinstance(model_ft, AlexNet_LwF):
        model_ft.model.classifier = nn.Sequential(
            *list(model_ft.model.classifier.children())[:model_ft.last_layer_name + 1])
        model_ft = model_ft.model
    elif isinstance(model_ft, AlexNet_EBLL):
        model_ft.classifier = nn.Sequential(*list(model_ft.classifier.children())[:model_ft.last_layer_name + 1])
        model_ft.set_finetune_mode(True)

    if freeze_mode or replace_last_classifier_layer:
        labels_per_task = [len(task_labels) for task_labels in dset_classes['train']]
        output_labels = sum(labels_per_task)
        model_ft = utils.replace_last_classifier_layer(model_ft, output_labels)
        print("REPLACED LAST LAYER with {} new output nodes".format(output_labels))
    current_label_list = None
    if use_gpu:
        model_ft = model_ft.cuda()
        print("MODEL LOADED IN CUDA GPU")
    if args.class_incremental:
        if args.task_counter>1:

            tg_params = model_ft.parameters()
            optimizer_ft = optim.SGD(tg_params, lr=lr, momentum=0.9,weight_decay=weight_decay)
            if args.class_incremental and dset_class_to_idx is not None:
                order_root_path = os.path.dirname(model_path)
                order_name = os.path.join(order_root_path,
                                          "generator_label_list_order_seed={}.pkl".format(args.CI_order_rndseed))
                print("Order name:{}".format(order_name))
                print(dset_class_to_idx['train'])
                label_order_list = utils.unpickle(order_name)
                current_label_list = deepcopy(label_order_list)
                current_label_list.extend(dset_class_to_idx['train'][0])
                current_label_list.sort()
        else:
            tg_params = model_ft.parameters()
            optimizer_ft = optim.SGD(tg_params, lr=lr, momentum=0.9,
                                     weight_decay=weight_decay)
            if args.class_incremental and dset_class_to_idx is not None:
                print(dset_class_to_idx['train'])
                current_label_list = []
                current_label_list.extend(dset_class_to_idx['train'][0])
                current_label_list.sort()

    else:
        if freeze_mode:
            optimizer_ft = optim.SGD(model_ft.classifier._modules['6'].parameters(), lr, momentum=0.9)
        else:
            optimizer_ft = optim.SGD(model_ft.parameters(), lr, momentum=0.9, weight_decay=weight_decay)

    model_ft, best_acc = SGD_Training.train_model(model_ft, criterion, optimizer_ft, lr, dset_dataloader,
                                                  cumsum_dset_sizes, use_gpu, num_epochs, exp_dir, resume,
                                                  save_models_mode=save_models_mode,
                                                  saving_freq=freq,args=args,current_label_list=current_label_list)


    return model_ft, best_acc
def save_runningtime(time_elapsed,filename='runningtime.txt'):
    File = open(filename,'w')
    File.write(time_elapsed)
    File.close()
