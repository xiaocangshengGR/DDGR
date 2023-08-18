

import os


import torch
import shutil

import random
from torchvision import transforms

import src.utilities.utils as utils
from src.data.imgfolder import random_split, ImageFolderTrainVal
from copy import deepcopy
import numpy as np
def download_dset(path):
    print('Please download ImageNet into {} manually.'.format(path),
          'You also need to resize the images into 64*64, and store them into a new dir: imagenet_resized_64-1000.')




def divide_into_tasks(root_path, task_count=5,order_random_seed=None):



    print("Be patient: dividing into tasks...")
    nr_classes_remain = 1000 // 2
    nr_classes_per_task = 500 // task_count
    assert 500 % nr_classes_per_task == 0, "There are 500 classes divied into the first task, and the remaining 500 classes must be divisible by nb classes per task"

    file_path = os.path.join(root_path, "classes.txt")
    lines = [line.rstrip('\n') for line in open(file_path)]
    assert len(lines) == 1000, "Should have 1000 classes, but {} lines in classes.txt".format(len(lines))

    out_dir = os.path.join(root_path, "no_crop",
                           "imagenet1000CI_init_task+{}tasks_rndseed={}".format(task_count, order_random_seed))
    order_name = os.path.join(out_dir,
                              "order_seed={}.pkl".format(order_random_seed))
    print("Order name:{}".format(order_name))
    if order_random_seed is not None:
        np.random.seed(order_random_seed)
    if os.path.exists(order_name):
        print("Loading orders")
        order = utils.unpickle(order_name)
    else:
        print("Generating orders")
        if order_random_seed is None:
            order = deepcopy(lines)
        else:
            order0 = deepcopy(lines)
            order1 = order0[:nr_classes_remain]
            order2 = order0[nr_classes_remain:]
            np.random.shuffle(order2)
            order = []
            order.extend(order1)
            order.extend(order2)
        utils.savepickle(order, order_name)
    class_order_list = order

    np.random.seed(7)

    subsets = ['train', 'val']
    img_paths = {t: {s: [] for s in subsets + ['classes', 'class_to_idx']} for t in range(1, task_count + 2)}

    for subset in subsets:

        classes = class_order_list[0:nr_classes_remain]
        class_to_idx = {classes[i]: i for i in range(nr_classes_remain)}
        task = 1
        if len(img_paths[task]['classes']) == 0:
            img_paths[task]['classes'].extend(classes)
        img_paths[task]['class_to_idx'] = list(range(nr_classes_remain))
        for class_index in range(nr_classes_remain):
            target = class_order_list[class_index]
            src_path = os.path.join(root_path, subset, target)
            all_img_per_cls_list = os.listdir(src_path)
            chose_img_num_per_class = int(len(all_img_per_cls_list))
            random.seed(7)
            chose_img_per_class_list =random.sample(all_img_per_cls_list,chose_img_num_per_class)
            random.seed(7)
            imgs = [(os.path.join(src_path, f), class_to_idx[target]) for f in chose_img_per_class_list
                    if os.path.isfile(os.path.join(src_path, f))]
            img_paths[task][subset].extend(imgs)
        task = task + 1

        for initial_class in (range(nr_classes_remain, len(class_order_list), nr_classes_per_task)):
            classes = class_order_list[initial_class:initial_class + nr_classes_per_task]
            class_to_idx = {classes[i-initial_class]: i for i in range(initial_class,initial_class + nr_classes_per_task)}
            if len(img_paths[task]['classes']) == 0:
                img_paths[task]['classes'].extend(classes)
            img_paths[task]['class_to_idx'] = list(range(initial_class,initial_class + nr_classes_per_task))

            for class_index in range(initial_class, initial_class + nr_classes_per_task):
                target = class_order_list[class_index]
                src_path = os.path.join(root_path, subset, target)

                all_img_per_cls_list = os.listdir(src_path)
                chose_img_num_per_class = int(len(all_img_per_cls_list))
                random.seed(7)
                chose_img_per_class_list = random.sample(all_img_per_cls_list, chose_img_num_per_class)
                random.seed(7)

                imgs = [(os.path.join(src_path, f), class_to_idx[target]) for f in chose_img_per_class_list
                        if os.path.isfile(os.path.join(src_path, f))]
                img_paths[task][subset].extend(imgs)
            task = task + 1
    return img_paths


def create_train_test_val_imagefolders(img_paths, root, normalize, include_rnd_transform, no_crop):

    pre_transf = None
    if include_rnd_transform:
        if no_crop:
            pre_transf = transforms.RandomHorizontalFlip()
        else:
            pre_transf = transforms.Compose([
                transforms.RandomResizedCrop(56),
                transforms.RandomHorizontalFlip(), ])
    else:
        if not no_crop:
            pre_transf = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(56),
            ])
    sufx_transf = [transforms.ToTensor(), normalize, transforms.Resize(256),]

    train_transf = transforms.Compose([pre_transf] + sufx_transf) if pre_transf else transforms.Compose(sufx_transf)
    train_dataset = ImageFolderTrainVal(root, None, transform=train_transf, classes=img_paths['classes'],
                                        class_to_idx=img_paths['class_to_idx'], imgs=img_paths['train'])


    pre_transf_val = None
    sufx_transf_val = [transforms.ToTensor(), normalize, transforms.Resize(256),]

    val_transf = transforms.Compose([pre_transf_val] + sufx_transf_val) if pre_transf_val \
        else transforms.Compose(sufx_transf_val)
    test_dataset = ImageFolderTrainVal(root, None, transform=val_transf, classes=img_paths['classes'],
                                       class_to_idx=img_paths['class_to_idx'], imgs=img_paths['val'])

    dsets = {}
    dsets['train'] = train_dataset
    dsets['test'] = test_dataset

    dset_trainval = random_split(dsets['train'],
                                 [round(len(dsets['train']) * (0.8)), round(len(dsets['train']) * (0.2))])
    dsets['train'] = dset_trainval[0]
    dsets['val'] = dset_trainval[1]
    dsets['val'].transform = val_transf
    print("Created Dataset:{}".format(dsets))
    return dsets


def create_train_val_test_imagefolder_dict(dataset_root, img_paths, task_count, outfile, no_crop=True, transform=False,
                                           order_random_seed=None):

    if no_crop:
        out_dir = os.path.join(dataset_root, "no_crop",
                               "imagenet1000CI_init_task+{}tasks_rndseed={}".format(task_count, order_random_seed))
    else:
        out_dir = os.path.join(dataset_root,
                               "imagenet1000CI_init_task+{}tasks_rndseed={}".format(task_count, order_random_seed))

    for task in range(1, task_count + 2):
        print("\nTASK ", task)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        dsets = create_train_test_val_imagefolders(img_paths[task], dataset_root, normalize, transform, no_crop)
        utils.create_dir(os.path.join(out_dir, str(task)))
        torch.save(dsets, os.path.join(out_dir, str(task), outfile))
        print("SIZES: train={}, val={}, test={}".format(len(dsets['train']), len(dsets['val']),
                                                        len(dsets['test'])))
        print("Saved dictionary format of train/val/test dataset Imagefolders.")


def create_train_val_test_imagefolder_dict_joint(dataset_root, img_paths, outfile, no_crop=True):

    if no_crop:
        out_dir = os.path.join(dataset_root, "no_crop")
    else:
        out_dir = dataset_root

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dsets = create_train_test_val_imagefolders(img_paths[1], dataset_root, normalize, True, no_crop=no_crop)

    utils.create_dir(out_dir)
    torch.save(dsets, os.path.join(out_dir, outfile))
    print("JOINT SIZES: train={}, val={}, test={}".format(len(dsets['train']), len(dsets['val']),
                                                          len(dsets['test'])))
    print("JOINT: Saved dictionary format of train/val/test dataset Imagefolders.")


def prepare_dataset(dset, target_path, survey_order=True, joint=True, task_count=10, overwrite=False,order_random_seed=None):
    print("Preparing dataset")
    if not os.path.isdir(target_path):
        raise Exception("IMGNET PATH IS NON EXISTING DIR: ", target_path)

    if os.path.isdir(os.path.join(target_path, 'train')):
        shutil.copyfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "imgnet_classes.txt"),
                            os.path.join(target_path, 'classes.txt'))
    else:
        print("Already cleaned up original train")

    if not os.path.isfile(os.path.join(target_path, 'VAL_PREPROCESS.TOKEN')):
        torch.save({}, os.path.join(target_path, 'VAL_PREPROCESS.TOKEN'))
    else:
        print("Already cleaned up original val")
    out_dir = os.path.join(target_path, "no_crop",
                           "imagenet1000CI_init_task+{}tasks_rndseed={}".format(task_count, order_random_seed))
    if not os.path.isfile(os.path.join(out_dir, "DIV.TOKEN")) or overwrite:
        print("PREPARING DATASET: DIVIDING INTO {} TASKS".format(task_count+1))
        img_paths = divide_into_tasks(target_path, task_count=task_count,order_random_seed=order_random_seed)
        torch.save({}, os.path.join(out_dir, 'DIV.TOKEN'))
    else:
        print("Already divided into tasks")

    if not os.path.isfile(os.path.join(out_dir, "IMGFOLDER.TOKEN")) or overwrite:
        print("PREPARING DATASET: IMAGEFOLDER GENERATION")
        create_train_val_test_imagefolder_dict(target_path, img_paths, task_count, dset.raw_dataset_file,
                                               no_crop=True, transform=False,order_random_seed=order_random_seed)
        create_train_val_test_imagefolder_dict(target_path, img_paths, task_count, dset.transformed_dataset_file,
                                               no_crop=True, transform=True,order_random_seed=order_random_seed)
        torch.save({}, os.path.join(out_dir, 'IMGFOLDER.TOKEN'))
    else:
        print("Task imgfolders already present.")

    print("PREPARED DATASET")
