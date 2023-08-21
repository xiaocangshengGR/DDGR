
import torch
import numpy as np
import cv2

from torch.utils.data import TensorDataset, Dataset, Subset, DataLoader
import pickle
from copy import deepcopy
import os
import zipfile
from PIL import Image
import urllib.request
from torchvision import transforms
import gc

import src.utilities.utils as utils
from src.data.core50CIR_data_loader import CORE50
from src.data.imgfolder import ImageFolderTrainVal

transform = transforms.Compose([transforms.ToTensor()])
class SplitDataSet(Dataset):

    def __init__(self, dataset_x, dataset_y, split_test_label_list=[]):
        self.dataset_x = deepcopy(dataset_x)
        self.dataset_y = deepcopy(dataset_y)

        if len(split_test_label_list) == 0:
            self.classes = list(set(self.dataset_y.squeeze()))
            self.classes.sort()
            self.class_to_idx = deepcopy(self.classes)
            self.x = self.dataset_x
            self.y = self.dataset_y
            self.dataset_len = len(self.y)
        else:
            self.classes = split_test_label_list
            self.classes.sort()
            self.class_to_idx = deepcopy(self.classes)
            task_idx = torch.nonzero(torch.from_numpy(
                np.isin(self.dataset_y, split_test_label_list))).squeeze()
            self.x = self.dataset_x[task_idx]
            self.y = self.dataset_y[task_idx]

            self.dataset_len = len(self.y)

    def __getitem__(self, index):
        xx=transform(self.x[index])
        img, target = xx, self.y[index]
        return img, target

    def __len__(self):
        return len(self.y)

    def get_xy(self):
        return self.x,self.y

def download(url, path):
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = os.path.join(path, url.split("/")[-1])

    if os.path.exists(file_name):
        print("Dataset already downloaded at {}.".format(file_name))
    else:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Wget/1.20.3 (linux-gnu)')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, file_name, ProgressBar().update)

    return file_name


def unzip(path):
    directory_path = os.path.dirname(path)

    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(directory_path)


class ProgressBar:

    def __init__(self):
        self.count = 0

    def update(self, tmp, block_size, total_size):
        self.count += block_size

        percent = "{}".format(int(100 * self.count / total_size))
        filled_length = int(100 * self.count // total_size)
        pbar = "#" * filled_length + '-' * (100 - filled_length)

        print("\r|%s| %s%%" % (pbar, percent), end="\r")
        if self.count == total_size:
            print()




def dump_data(obj,p):
    f = open(p,'wb')
    pickle.dump(obj,f,protocol=4)
    f.close()

def load_data(p):
    f = open(p,'rb')
    data = pickle.load(f)
    f.close()
    return data

def get_dset(data_dir,num_run=0):

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    label_url = 'https://vlomonaco.github.io/core50/data/labels.pkl'
    LUP_url = 'https://vlomonaco.github.io/core50/data/LUP.pkl'
    img_url = 'http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz'
    path_url = 'https://vlomonaco.github.io/core50/data/paths.pkl'

    out_train = os.path.join(data_dir, 'core50CIREP')
    if not os.path.exists(out_train):

        utils.create_dir(out_train)
        print(os.path.join(out_train, "core50_imgs.npz"))
        if os.path.exists(os.path.join(out_train, "core50_imgs.npz")):
            print("core50_imgs.npz already downloaded.")
        else:
            print("Downloading the core50_imgs.npz...")
            download(img_url, out_train)

        if os.path.exists(os.path.join(out_train, "labels.pkl")):
            print("labels.pkl already downloaded.")
        else:
            print("Downloading the labels.pkl...")
            download(label_url, out_train)

        if os.path.exists(os.path.join(out_train, "LUP.pkl")):
            print("LUP.pkl already downloaded.")
        else:
            print("Downloading the LUP.pkl...")
            download(LUP_url, out_train)
        if os.path.exists(os.path.join(out_train, "paths.pkl")):
            print("paths.pkl already downloaded.")
        else:
            print("Downloading the paths.pkl...")
            download(path_url, out_train)

        print("Succesfully download core50_imgs.npz, labels.pkl, LUP.pkl, paths.pkl.")
    else:
        print("Already download core50_imgs.npz, labels.pkl, LUP.pkl, paths.pkl in {}".format(os.path.join(data_dir, 'core50CIREP')))


def divide_into_tasks(root_path,dataset,fix_test_x,fix_test_y,task_count=79,num_run=None,outfile='None.pth.tar',CI_order_rndseed=None):

    fix_test_y = np.array(fix_test_y,dtype=int)
    print("Be patient: dividing into tasks...")

    out_dir = os.path.join(root_path, "no_crop", "core50CIREP_{}tasks_num_run={}".format(task_count,num_run))
    order_name = os.path.join(out_dir,
                              "order_seed={}.pkl".format(CI_order_rndseed))
    class_order_list = list(range(50))

    utils.savepickle(class_order_list, order_name)

    subsets = ['train', 'val', 'test', 'fixtest']

    sufx_transf = [transforms.ToTensor(), transforms.Resize(128), ]
    train_transf = transforms.Compose(sufx_transf)

    fix_test_dir = os.path.join(out_dir,"fix_test_data")
    fix_test_data = SplitDataSet(fix_test_x, fix_test_y)
    fix_test_filepath_label_list = save_to_JPEG(fix_test_x,fix_test_y,fix_test_dir,0)

    np.random.seed(7)
    num_img = 0
    for i, train_batch in enumerate(dataset):
        task = i + 1
        print(task)
        temp_dict={}
        d_x, d_y = train_batch
        d_y = np.array(d_y,dtype=int)
        index = np.arange(len(d_y))
        np.random.shuffle(index)
        index_list = list(index)
        train_num = int(len(d_y)*0.8)
        shuffle_x = d_x[index_list]
        shuffle_y = d_y[index_list]
        task_data_dir = os.path.join(out_dir,"{}".format(task))
        for subset in subsets:
            if subset == 'fixtest':
                temp_dict[subset] = ImageFolderTrainVal(os.path.join(task_data_dir,subset), None,
                                                        transform=train_transf, classes=fix_test_data.classes,
                                                        class_to_idx=fix_test_data.class_to_idx,
                                                        imgs=fix_test_filepath_label_list)
            elif subset == 'train':
                train_x = shuffle_x[0:train_num]
                train_y = shuffle_y[0:train_num]
                dd = SplitDataSet(train_x,train_y)
                filepath_label_list = save_to_JPEG(train_x,train_y,os.path.join(task_data_dir,subset),num_img)
                temp_dict[subset] = ImageFolderTrainVal(os.path.join(task_data_dir, subset), None,
                                                        transform=train_transf,classes=dd.classes,
                                                        class_to_idx=dd.class_to_idx,imgs=filepath_label_list)
                num_img+=len(train_y)
            elif subset == 'val':
                val_x = shuffle_x[train_num:]
                val_y = shuffle_y[train_num:]
                dd = SplitDataSet(val_x,val_y)
                filepath_label_list = save_to_JPEG(val_x,val_y,os.path.join(task_data_dir,subset),num_img)
                temp_dict[subset] = ImageFolderTrainVal(os.path.join(task_data_dir, subset), None,
                                                        transform=train_transf, classes=dd.classes,
                                                        class_to_idx=dd.class_to_idx, imgs=filepath_label_list)
                num_img += len(val_y)
            else:
                current_task_test_labels = list(set(d_y.squeeze()))
                current_task_test_labels.sort()
                dd = SplitDataSet(fix_test_x,fix_test_y,current_task_test_labels)
                sfix_x,sfix_y = dd.get_xy()
                filepath_label_list = save_to_JPEG(sfix_x,sfix_y,os.path.join(task_data_dir,subset),0)
                temp_dict[subset] = ImageFolderTrainVal(os.path.join(task_data_dir, subset), None,
                                                        transform=train_transf, classes=dd.classes,
                                                        class_to_idx=dd.class_to_idx, imgs=filepath_label_list)
                temp_dict['classes'] = temp_dict[subset].classes
                temp_dict['class_to_idx'] = temp_dict[subset].class_to_idx
        torch.save(temp_dict,os.path.join(task_data_dir, outfile),pickle_protocol=4)
        del temp_dict
        gc.collect()
    np.random.seed(7)

def save_to_JPEG(images_list,labels_list,output_path,num_img):

    return_list = []
    if not os.path.exists(output_path):
        utils.create_dir(output_path)
        counter = 0
        for i in range(len(images_list)):
            img = images_list[i]
            label = labels_list[i]
            filename = os.path.join(output_path,"{}_{}.JPEG".format(label,num_img+counter))
            cv2.imwrite(filename, img)

            img = Image.open(filename)
            img = img.convert('RGB')
            resized_img = img.resize((64, 64), Image.BILINEAR)
            resized_img.save(filename)

            return_list.append((filename,label))

            counter+=1
    else:
        img_list = os.listdir(output_path)
        for img in img_list:
            p = img
            l = int(p.split('_')[0])
            return_list.append((os.path.join(output_path,p),l))

    return return_list

def prepare_dataset(dset,
                    target_path, task_count=10, overwrite=False, num_run = 0,CI_order_rndseed=None):

    print("Preparing dataset")
    if not os.path.isdir(target_path):
        raise Exception("core50CIREP PATH IS NON EXISTING DIR: ", target_path)

    out_dir = os.path.join(target_path, "no_crop",
                           "core50CIREP_{}tasks_num_run={}".format(task_count, num_run))
    if not os.path.isfile(os.path.join(out_dir, "DIV.TOKEN")) or overwrite:
        print("PREPARING DATASET: DIVIDING INTO {} TASKS".format(task_count))
        dataset = CORE50(root=target_path,
                         preload=True, scenario="nic", run=num_run)
        fix_test_x, fix_test_y = dataset.get_test_set()
        print('Get dataset, fix_test_x and fix_test_y.')
        divide_into_tasks(target_path, dataset,fix_test_x,fix_test_y, task_count=task_count,
                                      num_run=num_run,outfile=dset.raw_dataset_file,CI_order_rndseed=CI_order_rndseed)
        torch.save({}, os.path.join(out_dir, 'DIV.TOKEN'))
    else:
        print("Already divided into tasks")

    print("PREPARED DATASET")
