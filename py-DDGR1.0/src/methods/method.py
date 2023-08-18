from abc import ABC, abstractmethod
from enum import Enum, auto
from collections import OrderedDict
import os

import copy

import src.utilities.utils as utils
import torch
from torch.autograd import Variable

import src.utilities.utils

from src.data.imgfolder import ConcatDatasetDynamicLabels

import src.framework.inference as test_network

import src.methods.DDGR.main_DDGR as trainDDGR
import src.methods.Finetune.main_SGD as trainFT


def parse(method_name):

    if method_name == DDGR.name:
        return DDGR()

    elif method_name == Finetune.name:
        return Finetune()

    else:
        raise NotImplementedError("Method not yet parseable")


class Method(ABC):
    @property
    @abstractmethod
    def name(self): pass

    @property
    @abstractmethod
    def eval_name(self): pass

    @property
    @abstractmethod
    def category(self): pass

    @property
    @abstractmethod
    def extra_hyperparams_count(self): pass

    @property
    @abstractmethod
    def hyperparams(self): pass

    @classmethod
    def __subclasshook__(cls, C):
        return False

    @abstractmethod
    def get_output(self, images, args): pass

    @staticmethod
    @abstractmethod
    def inference_eval(args, manager): pass


class Category(Enum):
    MODEL_BASED = auto()
    DATA_BASED = auto()
    MASK_BASED = auto()
    BASELINE = auto()
    REHEARSAL_BASED = auto()

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value



def get_output_def(model, heads, images, current_head_idx, final_layer_idx,dset_classes=None):

    if not (heads == None):
        head = heads[current_head_idx]
        model.classifier._modules[final_layer_idx] = head
    model.eval()
    if dset_classes is not None:
        outputs = model(Variable(images),dset_classes)
    else:
        outputs = model(Variable(images))
    return outputs


def set_hyperparams(method, hyperparams, static_params=False):

    assert isinstance(hyperparams, str)
    leave_default = lambda x: x == 'def' or x == ''
    hyperparam_vals = []
    split_lists = [x.strip() for x in hyperparams.split('_') if len(x) > 0]
    for split_list in split_lists:
        split_params = []
        for x in split_list.split(','):
            if not leave_default(x):
                if 'False' in x:
                    split_params.append(False)
                elif 'True' in x:
                    split_params.append(True)
                else:
                    split_params.append(float(x))
        split_params = split_params[0] if len(split_params) == 1 else split_params
        if len(split_lists) == 1:
            hyperparam_vals.append(split_params)
        else:
            hyperparam_vals.append(split_params)
    if static_params:
        if not hasattr(method, 'static_hyperparams'):
            print("No static hyperparams to set.")
            return
        target = method.static_hyperparams
    else:
        target = method.hyperparams

    for hyperparam_idx, (hyperparam_key, def_val) in enumerate(target.items()):
        if hyperparam_idx < len(hyperparam_vals):
            arg_val = hyperparam_vals[hyperparam_idx]
            if leave_default(arg_val):
                continue
            if 'IF_order' in hyperparam_key:
                print(hyperparam_key)
                print(arg_val)
                arg_val = int(arg_val)
            target[hyperparam_key] = arg_val
            print("Set value {}={}".format(hyperparam_key, target[hyperparam_key]))
        else:
            print("Retaining default value {}={}".format(hyperparam_key, def_val))
    method.init_hyperparams = copy.deepcopy(target)
    print("INIT HYPERPARAMETERS: {}".format(target))

class DDGR(Method):
    name = "DDGR"
    eval_name = name
    category = Category.MODEL_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'ulhyper': 3.0})
    static_hyperparams = OrderedDict({'classifier_type': 'self'})
    wrap_first_task_model = False
    start_scratch = True
    @staticmethod
    def grid_train(args, manager, lr):
        return Finetune.grid_train(args, manager, lr)

    def train(self, args, manager,hyperparams):

        model_ft, best_acc = trainDDGR.fine_tune_train_DDGR(dataset_path=manager.current_task_dataset_path,
                                                              args=args,
                                                              previous_task_model_path=manager.previous_task_model_path,
                                                              exp_dir=manager.heuristic_exp_dir,
                                                              task_counter=args.task_counter - 1,
                                                              batch_size=args.batch_size,
                                                              num_epochs=args.num_epochs,
                                                              lr=args.lr,
                                                              weight_decay=args.weight_decay,
                                                              head_shared=False if not ('alexnet' in args.model_name or
                                                                          args.model_name == 'PermutedMLP' or args.class_incremental
                                                                          or args.class_incremental_repetition) else True,
                                                              saving_freq=args.saving_freq,
                                                              classifier_type=self.static_hyperparams['classifier_type']
                                                              )
        self.task_counter = args.task_counter
        return model_ft, best_acc

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx,
                              args.dset_classes)

    @staticmethod
    def inference_eval(args, manager, string_name):
        return Finetune.inference_eval(args, manager, string_name)

class Finetune(Method):
    name = "finetuning"
    eval_name = name
    category = Category.BASELINE
    extra_hyperparams_count = 0
    hyperparams = {}
    grid_chkpt = True
    start_scratch = True
    no_framework = True

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx)

    @staticmethod
    def grid_train(args, manager, lr):
        dataset_path = manager.current_task_dataset_path
        print('lr is ' + str(lr))
        print("DATASETS: ", dataset_path)

        if not isinstance(dataset_path, list):
            dataset_path = [dataset_path]

        dset_dataloader, cumsum_dset_sizes, dset_classes,dset_class_to_idx = Finetune.compose_dataset(dataset_path, args.batch_size)
        return trainFT.fine_tune_SGD(dset_dataloader, cumsum_dset_sizes, dset_classes,
                                     model_path=manager.previous_task_model_path,
                                     exp_dir=manager.gridsearch_exp_dir,
                                     num_epochs=args.num_epochs, lr=lr,
                                     weight_decay=args.weight_decay,
                                     enable_resume=True,
                                     save_models_mode=True,
                                     freq=args.saving_freq,
                                     method_name=args.method_name,args=args,
                                     replace_last_classifier_layer = False if args.model_name=='PermutedMLP'
                                                                              or args.class_incremental
                                                                              or args.class_incremental_repetition else True,
                                     dset_class_to_idx=dset_class_to_idx
                                     )

    @staticmethod
    def grid_poststep(args, manager):
        manager.previous_task_model_path = os.path.join(manager.best_exp_grid_node_dirname, 'best_model.pth.tar')

        print("SINGLE_MODEL MODE: Set previous task model to ", manager.previous_task_model_path)
        Finetune.grid_poststep_symlink(args, manager)

    @staticmethod
    def grid_poststep_symlink(args, manager):
        exp_dir = os.path.join(manager.parent_exp_dir, 'task_' + str(args.task_counter), 'TASK_TRAINING')
        if os.path.exists(exp_dir):
            os.unlink(exp_dir)
        print("Symlink best LR: ", src.utilities.utils.get_relative_path(manager.best_exp_grid_node_dirname, segments=2))
        os.symlink(src.utilities.utils.get_relative_path(manager.best_exp_grid_node_dirname, segments=2), exp_dir)

    @staticmethod
    def compose_dataset(dataset_path, batch_size):
        dset_imgfolders = {x: [] for x in ['train', 'val']}
        dset_classes = {x: [] for x in ['train', 'val']}
        dset_sizes = {x: [] for x in ['train', 'val']}
        dset_class_to_idx = {x: [] for x in ['train', 'val']}

        for dset_count in range(0, len(dataset_path)):
            dset_wrapper = torch.load(dataset_path[dset_count])

            for mode in ['train', 'val']:
                dset_imgfolders[mode].append(dset_wrapper[mode])
                dset_classes[mode].append(dset_wrapper[mode].classes)
                dset_sizes[mode].append(len(dset_wrapper[mode]))
                dset_class_to_idx[mode].append(dset_wrapper[mode].class_to_idx)

        cumsum_dset_sizes = {mode: sum(dset_sizes[mode]) for mode in dset_sizes}
        classes_len = {mode: [len(ds) for ds in dset_classes[mode]] for mode in dset_classes}
        dset_dataloader = {x: torch.utils.data.DataLoader(
            ConcatDatasetDynamicLabels(dset_imgfolders[x], classes_len[x]),
            batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
            for x in ['train', 'val']}
        print("dset_classes: {}, dset_sizes: {}".format(dset_classes, cumsum_dset_sizes))
        return dset_dataloader, cumsum_dset_sizes, dset_classes,dset_class_to_idx

    @staticmethod
    def inference_eval(args, manager, string_name):

        if not args.class_incremental:
            print("EVAL on prev model: ",args.eval_model_path)
        model = torch.load(args.eval_model_path)
        if args.class_incremental and (args.method_name=='DDGR'):
            order_name = os.path.join(os.path.dirname(args.eval_model_path),
                                      "generator_label_list_order_seed={}.pkl".format(args.CI_order_rndseed))
            if not args.class_incremental:
                print("Order name:{}".format(order_name))
            label_order_list = utils.unpickle(order_name)
        else:
            label_order_list = None
        if isinstance(model, dict):
            model = model['model']
        if args.model_name == 'PermutedMLP' or args.class_incremental or args.class_incremental_repetition:
            target_heads = None
            target_head_idx = 0
        else:
            head_layer_idx = str(len(model.classifier._modules) - 1)
            current_head = model.classifier._modules[head_layer_idx]
            assert isinstance(current_head, torch.nn.Linear), "NO VALID HEAD IDX"
            target_heads = src.utilities.utils.get_prev_heads(args.head_paths, head_layer_idx)
            target_head_idx = 0
            assert len(target_heads) == 1

        accuracy,s = test_network.test_model(manager.method, model, args.dset_path, target_head_idx, subset=args.test_set,
                                           target_head=target_heads, batch_size=args.batch_size,args=args,
                                           task_idx=args.eval_dset_idx,string_name=string_name,
                                           class_incremental=args.class_incremental or args.class_incremental_repetition,
                                           label_order_list=label_order_list)
        return accuracy,s
