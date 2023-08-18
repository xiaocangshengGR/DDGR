

from torchvision import datasets

import torch
import numpy as np

from torch.utils.data import TensorDataset, Dataset, Subset
import pickle
from copy import deepcopy
import os
from PIL import Image
from torchvision import transforms

import src.utilities.utils as utils
from src.data.imgfolder import random_split, ImageFolderTrainVal
CIFAR100_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
def transfunc(is_train):
    if is_train:
        trans = [
                 transforms.Resize((64, 64)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[CIFAR100_mean[0],CIFAR100_mean[1],CIFAR100_mean[2]],
                                      std=[CIFAR100_std[0], CIFAR100_std[1],CIFAR100_std[2]])]
        trans = transforms.Compose(trans)

    else:
        trans = [transforms.Resize((64, 64)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[CIFAR100_mean[0],CIFAR100_mean[1],CIFAR100_mean[2]],
                                      std=[CIFAR100_std[0], CIFAR100_std[1],CIFAR100_std[2]])]
        trans = transforms.Compose(trans)
    return trans

class SplitCIFAR100:


    def __init__(self, train_dataset, val_dataset, task_count, class_order_list):
        l = int(len(val_dataset)/2)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = deepcopy(val_dataset)

        self.nr_classes = 100
        self.nr_classes_remain = int(self.nr_classes / 2)
        self.max_iter = task_count + 1
        assert self.nr_classes_remain % task_count == 0, "class_num={} % task_count={} != 0".format(self.nr_classes_remain,
                                                                                                task_count)
        self.nr_classes_per_task = int(self.nr_classes_remain / task_count)

        self.class_order_list = class_order_list
        self.cur_iter = 0
        self.class_sets = []
        for it in range(self.max_iter):
            if it == 0:
                self.class_sets.append(self.class_order_list[0:self.nr_classes_remain])
            else:
                self.class_sets.append(self.class_order_list[self.nr_classes_remain + (it-1)*self.nr_classes_per_task: self.nr_classes_remain + it*self.nr_classes_per_task])


    def get_dims(self):

        return len(self.train_dataset) / self.nr_classes_per_task, self.nr_classes_per_task

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            train_dataset = SplitDataSet(self.train_dataset, self.cur_iter, self.nr_classes,
                                         self.nr_classes_per_task,self.class_order_list)
            val_dataset = SplitDataSet(self.val_dataset, self.cur_iter, self.nr_classes,
                                       self.nr_classes_per_task,self.class_order_list)
            test_dataset = SplitDataSet(self.test_dataset, self.cur_iter, self.nr_classes,
                                       self.nr_classes_per_task,self.class_order_list)
            self.cur_iter += 1

        return train_dataset, val_dataset,test_dataset, self.class_sets[self.cur_iter-1]


class SplitDataSet(Dataset):

    def __init__(self, dataset, cur_iter, nr_classes, nr_classes_per_task,class_order_list):
        self.dataset = dataset
        self.cur_iter = cur_iter
        self.nr_classes_remain = int(nr_classes/2)
        if self.cur_iter == 0:
            self.classes = class_order_list[0:self.nr_classes_remain]
            self.class_to_idx = list(range(self.nr_classes_remain))
        else:
            self.classes = class_order_list[self.nr_classes_remain+nr_classes_per_task*(self.cur_iter-1):self.nr_classes_remain+nr_classes_per_task*self.cur_iter]
            self.class_to_idx = list(range(self.nr_classes_remain+nr_classes_per_task*(self.cur_iter-1),self.nr_classes_remain+nr_classes_per_task*self.cur_iter))
        self.classes_all = class_order_list

        targets = self.dataset.targets
        if self.cur_iter == 0:
            task_idx = torch.nonzero(torch.from_numpy(
                np.isin(targets, self.classes_all[0:self.nr_classes_remain])))
        else:
            task_idx = torch.nonzero(torch.from_numpy(
                np.isin(targets, self.classes_all[self.nr_classes_remain+nr_classes_per_task * (self.cur_iter-1):
                                                           self.nr_classes_remain+nr_classes_per_task * self.cur_iter])))

        self.subset = Subset(self.dataset, task_idx)

    def __getitem__(self, index):
        img, target = self.subset[index]
        target = self.classes_all.index(target)
        return img, target

    def get_item(self,index):
        img, target = self.subset[index]
        target = self.classes_all.index(target)
        return img, target

    def __len__(self):
        return len(self.subset)
def dump_data(obj,path,par):
    if par == 'train':
        p = os.path.join(path,'train.pkl')
    elif par == 'val':
        p = os.path.join(path,'val.pkl')
    elif par == 'test':
        p = os.path.join(path,'test.pkl')
    f = open(p,'wb')
    pickle.dump(obj,f,2)
    f.close()

def load_data(p):
    f = open(p,'rb')
    data = pickle.load(f)
    f.close()
    return data
def get_dset(path):
    utils.create_dir(path)
    cifar100_train = datasets.CIFAR100(path,
                                       train=True, transform=None, download=True)
    cifar100_test = datasets.CIFAR100(path,
                                      train=False, transform=None, download=True)
    if not os.path.exists(os.path.join(path, 'cifar100CI')):
        out_train = os.path.join(path, 'cifar100CI', 'train', 'images')
        out_val = os.path.join(path, 'cifar100CI', 'val', 'images')
        out_test = os.path.join(path, 'cifar100CI', 'test', 'images')
        utils.create_dir(out_train)
        utils.create_dir(out_val)
        utils.create_dir(out_test)

        print("Succesfully extracted Cifar dataset.")
    else:
        print("Already extracted Mnist dataset in {}".format(os.path.join(path, 'cifar100CI')))
    return cifar100_train, cifar100_test

def create_training_classes_file(root_path):

    with open(os.path.join(root_path, 'classes.txt'), 'w') as classes_file:
        for class_dir in utils.get_immediate_subdirectories(os.path.join(root_path, 'train')):
            classes_file.write(class_dir + "\n")



def create_train_test_val_imagefolders(task,cifa100_datagen,
                                       out_dir, normalize,num_img):
    dsets = {}
    task_out_dir = os.path.join(out_dir,str(task))
    train_dataset, val_dataset, test_dataset, _ = cifa100_datagen.next_task()

    train_pl,train_count = save_to_JPEG(train_dataset,os.path.join(task_out_dir,"train"),num_img)
    val_pl, val_count = save_to_JPEG(val_dataset, os.path.join(task_out_dir, "val"), num_img)
    test_pl, _ = save_to_JPEG(test_dataset, os.path.join(task_out_dir, "test"), num_img)

    dsets['train'] = ImageFolderTrainVal(os.path.join(task_out_dir,"train"), None,
                                                        transform=normalize, classes=train_dataset.classes,
                                                        class_to_idx=train_dataset.class_to_idx,
                                                        imgs=train_pl)
    dsets['val'] = ImageFolderTrainVal(os.path.join(task_out_dir,"val"), None,
                                                        transform=normalize, classes=val_dataset.classes,
                                                        class_to_idx=val_dataset.class_to_idx,
                                                        imgs=val_pl)
    dsets['test'] = ImageFolderTrainVal(os.path.join(task_out_dir,"test"), None,
                                                        transform=normalize, classes=test_dataset.classes,
                                                        class_to_idx=test_dataset.class_to_idx,
                                                        imgs=test_pl)

    return dsets,train_count+val_count

def save_to_JPEG(datasets,output_path,num_img):

    return_list = []
    counter = 0
    if not os.path.exists(output_path):
        utils.create_dir(output_path)

        for i in range(len(datasets)):
            img, label = datasets.get_item(i)
            filename = os.path.join(output_path,"{}_{}.JPEG".format(label,num_img+counter))
            resized_img = img.resize((64, 64), Image.BILINEAR)
            resized_img.save(filename)
            return_list.append((filename,label))
            counter+=1
    else:
        img_list = os.listdir(output_path)
        for img in img_list:
            p = img
            l = int(p.split('_')[0])
            return_list.append((os.path.join(output_path, p), l))
            counter+=1

    return return_list,counter

def create_train_val_test_imagefolder_dict(dataset_root, task_count, outfile,cifar100CI_train,cifar100CI_test,
                                           no_crop=True, transform=False, order_random_seed=None):
    if no_crop:
        out_dir = os.path.join(dataset_root, "no_crop", "cifar100CI_init_task+{}tasks_rndseed={}".format(task_count,order_random_seed))
    else:
        out_dir = os.path.join(dataset_root, "cifar100CI_init_task+{}tasks_rndseed={}".format(task_count,order_random_seed))
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
            order = np.arange(100)
        else:
            order = np.arange(100)
            np.random.shuffle(order)
        utils.savepickle(order, order_name)
    class_order_list = list(order)

    np.random.seed(7)

    cifa100_datagen = SplitCIFAR100(cifar100CI_train, cifar100CI_test,task_count,class_order_list)

    num_img = 0
    for task in range(1, task_count + 2):
        print("\nTASK ", task)
        utils.create_dir(os.path.join(out_dir, str(task)))
        normalize =  transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
                                         transforms.Resize(64),])
        dsets,count_img = create_train_test_val_imagefolders(task,cifa100_datagen, out_dir, normalize, num_img)
        num_img+=count_img
        torch.save(dsets, os.path.join(out_dir, str(task), outfile))
        print("SIZES: train={}, val={}, test={}, num_img={}".format(len(dsets['train']), len(dsets['val']),
                                                        len(dsets['test']),num_img))

    print("Saved dictionary format of train/val/test dataset Imagefolders.")


def prepare_dataset(dset,cifar100CI_train,cifar100CI_test,
                    target_path, survey_order=True, joint=True, task_count=10, overwrite=False, order_random_seed = None):

    print("Preparing dataset")
    if not os.path.isdir(target_path):
        raise Exception("TINYIMGNET PATH IS NON EXISTING DIR: ", target_path)

    if not os.path.isdir(os.path.join(target_path, 'train')):
        print("Already cleaned up original train")

    if not os.path.isfile(os.path.join(target_path, 'VAL_PREPROCESS.TOKEN')):
        torch.save({}, os.path.join(target_path, 'VAL_PREPROCESS.TOKEN'))
    else:
        print("Already cleaned up original val")

    if not os.path.isfile(os.path.join(target_path, "DIV.TOKEN")) or overwrite:
        torch.save({}, os.path.join(target_path, 'DIV.TOKEN'))
    else:
        print("Already divided into tasks")
    out_dir = os.path.join(target_path, "no_crop", "cifar100CI_init_task+{}tasks_rndseed={}".format(task_count,order_random_seed))
    if not os.path.isfile(os.path.join(out_dir, "IMGFOLDER.TOKEN")) or overwrite:
        print("PREPARING DATASET: IMAGEFOLDER GENERATION")
        create_train_val_test_imagefolder_dict(target_path, task_count, dset.raw_dataset_file,
                                               cifar100CI_train, cifar100CI_test,
                                               no_crop=True, transform=False,order_random_seed=order_random_seed)
        torch.save({}, os.path.join(out_dir, 'IMGFOLDER.TOKEN'))
    else:
        print("Task imgfolders already present.")

    print("PREPARED DATASET")
