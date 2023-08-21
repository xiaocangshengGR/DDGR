import os
import time
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch

import src.utilities.utils as utils
import src.data.cifarCI_dataprep as dataprep_cifarCI
import src.data.imgnet1000CI_datapre as dataprep_imgnet1000CI
import src.data.core50CIR_dataprep as dataprep_core50CIR

def parse(ds_name,task_count=None,CI_order_rndseed=None,num_run=0):
    if ds_name == Cifar100ClassIncrementalDataset.argname:
        return  Cifar100ClassIncrementalDataset(task_count=task_count,CI_order_rndseed=CI_order_rndseed)
    elif ds_name == Imgnet1000ClassIncrementalDataset.argname:
        return Imgnet1000ClassIncrementalDataset(task_count=task_count,CI_order_rndseed=CI_order_rndseed)
    elif ds_name == Core50ClassIncrementalRepetitionDataset.argname:
        return Core50ClassIncrementalRepetitionDataset(task_count=task_count,num_run=num_run,CI_order_rndseed=CI_order_rndseed)
    else:
        raise NotImplementedError("Dataset not parseable: ", ds_name)


def get_nc_per_task(dataset):
    return [len(classes_for_task) for classes_for_task in dataset.classes_per_task.values()]


class CustomDataset(metaclass=ABCMeta):

    @property
    @abstractmethod
    def name(self): pass

    @property
    @abstractmethod
    def argname(self): pass

    @property
    @abstractmethod
    def test_results_dir(self): pass

    @property
    @abstractmethod
    def train_exp_results_dir(self): pass

    @property
    @abstractmethod
    def task_count(self): pass

    @property
    @abstractmethod
    def classes_per_task(self): pass

    @property
    @abstractmethod
    def input_size(self): pass

    @abstractmethod
    def get_task_dataset_path(self, task_name, rnd_transform):
        pass

    @abstractmethod
    def get_taskname(self, task_index):
        pass

class Cifar100ClassIncrementalDataset(CustomDataset):
    name = 'Cifar100CI'
    argname = 'cifar100CI'
    test_results_dir = 'cifar100CI'
    train_exp_results_dir = 'cifar100CI'
    def_task_count, task_count = 10, 10
    classes_per_task = OrderedDict()
    input_size = (32,32)

    def __init__(self, crop=False, create=True, task_count=10, dataset_root=None, overwrite=False,CI_order_rndseed=None):
        config = utils.get_parsed_config()

        self.dataset_root = dataset_root if dataset_root else os.path.join(
            utils.read_from_config(config, 'ds_root_path'), 'cifarCI', 'cifar100CI')
        print("Dataset root = {}".format(self.dataset_root))
        self.crop = crop
        self.task_count = task_count

        self.transformed_dataset_file = 'imgfolder_trainvaltest_rndtrans.pth.tar'
        self.raw_dataset_file = 'imgfolder_trainvaltest.pth.tar'
        self.joint_dataset_file = 'imgfolder_trainvaltest_joint.pth.tar'

        if create:
            cifar100CI_train,cifar100CI_test = dataprep_cifarCI.get_dset(os.path.dirname(self.dataset_root))
            dataprep_cifarCI.prepare_dataset(self, cifar100CI_train,cifar100CI_test, self.dataset_root,
                                           task_count=self.task_count, survey_order=True,
                                          overwrite=overwrite,order_random_seed=CI_order_rndseed)

        if not crop:
            self.dataset_root = os.path.join(self.dataset_root, 'no_crop')


        self.tasks_subdir = "_init_task+{}tasks_rndseed={}".format(task_count,CI_order_rndseed)

        self.test_results_dir += self.tasks_subdir
        self.train_exp_results_dir += self.tasks_subdir
        self.task_count += 1
        for task_name in range(1, self.task_count + 1):
            p = self.get_task_dataset_path(str(task_name))
            dsets = torch.load(p)
            dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
            dset_classes = dsets['train'].classes
            self.classes_per_task[str(task_name)] = dset_classes
            print("Task {}: dset_sizes = {}, #classes = {}".format(str(task_name), dset_sizes, len(dset_classes)))

    def get_task_dataset_path(self, task_name=None, rnd_transform=False):
        if task_name is None:
            return os.path.join(self.dataset_root, self.joint_dataset_file)

        filename = self.transformed_dataset_file if rnd_transform else self.raw_dataset_file
        return os.path.join(self.dataset_root, 'cifar100CI'+self.tasks_subdir, task_name, filename)

    def get_taskname(self, task_index):
        return str(task_index)



class Imgnet1000ClassIncrementalDataset(CustomDataset):
    name = 'Imagenet1000'
    argname = 'imagenet1000CI'
    test_results_dir = 'imagenet1000CI'
    train_exp_results_dir = 'imagenet1000CI'
    def_task_count, task_count = 5, 5
    classes_per_task = OrderedDict()
    input_size = (64, 64)

    def __init__(self, crop=False, create=True, task_count=5, dataset_root=None, overwrite=False,CI_order_rndseed=None):
        config = utils.get_parsed_config()

        self.dataset_root = dataset_root if dataset_root else os.path.join(
            utils.read_from_config(config, 'ds_root_path'), 'ImageNet', 'imagenet_resized_64-1000')
        print("Dataset root = {}".format(self.dataset_root))
        self.crop = crop
        self.task_count = task_count

        self.transformed_dataset_file = 'imgfolder_trainvaltest_rndtrans.pth.tar'
        self.raw_dataset_file = 'imgfolder_trainvaltest.pth.tar'
        self.joint_dataset_file = 'imgfolder_trainvaltest_joint.pth.tar'

        if create:
            dataprep_imgnet1000CI.download_dset(os.path.dirname(self.dataset_root))
            dataprep_imgnet1000CI.prepare_dataset(self, self.dataset_root, task_count=self.task_count, survey_order=True,
                                          overwrite=overwrite,order_random_seed=CI_order_rndseed)

        if not crop:
            self.dataset_root = os.path.join(self.dataset_root, 'no_crop')

        self.tasks_subdir = "_init_task+{}tasks_rndseed={}".format(task_count, CI_order_rndseed)

        self.test_results_dir += self.tasks_subdir
        self.train_exp_results_dir += self.tasks_subdir
        self.task_count += 1

        for task_name in range(1, self.task_count + 1):
            dsets = torch.load(self.get_task_dataset_path(str(task_name)))
            dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
            dset_classes = dsets['train'].classes
            self.classes_per_task[str(task_name)] = dset_classes
            print("Task {}: dset_sizes = {}, #classes = {}".format(str(task_name), dset_sizes, len(dset_classes)))

    def get_task_dataset_path(self, task_name=None, rnd_transform=False):
        if task_name is None:
            return os.path.join(self.dataset_root, self.joint_dataset_file)

        filename = self.transformed_dataset_file if rnd_transform else self.raw_dataset_file
        return os.path.join(self.dataset_root, 'imagenet1000CI'+self.tasks_subdir, task_name, filename)
    def get_taskname(self, task_index):
        return str(task_index)


class Core50ClassIncrementalRepetitionDataset(CustomDataset):
    name = 'Core50CIREP'
    argname = 'core50CIREP'
    test_results_dir = 'core50CIREP'
    train_exp_results_dir = 'core50CIREP'
    def_task_count, task_count = 79, 79
    classes_per_task = OrderedDict()
    input_size = (64,64)

    def __init__(self, crop=False, create=True, task_count=79, dataset_root=None, overwrite=False,num_run=0,CI_order_rndseed=None):
        config = utils.get_parsed_config()

        self.dataset_root = dataset_root if dataset_root else os.path.join(
            utils.read_from_config(config, 'ds_root_path'), 'core', 'core50CIREP')
        print("Dataset root = {}".format(self.dataset_root))
        self.crop = crop
        self.task_count = task_count

        self.transformed_dataset_file = 'imgfolder_trainvaltest_rndtrans.pth.tar'
        self.raw_dataset_file = 'imgfolder_trainvaltest.pth.tar'
        self.joint_dataset_file = 'imgfolder_trainvaltest_joint.pth.tar'

        if create:
            dataprep_core50CIR.get_dset(os.path.dirname(self.dataset_root),num_run)
            dataprep_core50CIR.prepare_dataset(self, self.dataset_root,
                                           task_count=self.task_count, overwrite=overwrite,num_run=num_run,CI_order_rndseed=CI_order_rndseed)

        if not crop:
            self.dataset_root = os.path.join(self.dataset_root, 'no_crop')

        self.tasks_subdir = "_{}tasks_num_run={}".format(task_count,num_run)

        self.test_results_dir += self.tasks_subdir
        self.train_exp_results_dir += self.tasks_subdir

    def get_task_dataset_path(self, task_name=None, rnd_transform=False):
        if task_name is None:
            return os.path.join(self.dataset_root, self.joint_dataset_file)

        filename = self.transformed_dataset_file if rnd_transform else self.raw_dataset_file
        return os.path.join(self.dataset_root, 'core50CIREP'+self.tasks_subdir, task_name, filename)

    def get_taskname(self, task_index):
        return str(task_index)