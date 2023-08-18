
from abc import ABCMeta, abstractmethod

import torch


from src.models.alexnet_fc7out import alexnetCI
import src.utilities.utils
import os


def parse_model_name(models_root_path, model_name, input_size,init_class_num=None):

    pretrained = "pretrained" in model_name
    if "alexnetCI" in model_name:
        base_model = AlexNetCI(models_root_path, pretrained=pretrained, create=True)
    else:
        raise NotImplementedError("MODEL NOT IMPLEMENTED YET: ", model_name)

    return base_model


def get_init_modelname(args):

    name = ["e={}".format(args.num_epochs),
            "bs={}".format(args.batch_size),
            "lr={}".format(sorted(args.lr_grid))]
    if args.weight_decay != 0:
        name.append("{}={}".format(ModelRegularization.weight_decay, args.weight_decay))
    if ModelRegularization.batchnorm in args.model_name:
        name.append(ModelRegularization.batchnorm)
    if ModelRegularization.dropout in args.model_name:
        name.append(ModelRegularization.dropout)
    return '_'.join(name)


def extract_modelname_val(seg, tr_exp_dir):
    seg_found = [tr_seg.split('=')[-1] for tr_seg in tr_exp_dir.split('_') if seg == tr_seg.split('=')[0]]
    if len(seg_found) == 1:
        return seg_found[0]
    elif len(seg_found) > 1:
        raise Exception("Ambiguity in exp name: {}".format(seg_found))
    else:
        return None


class ModelRegularization(object):
    vanilla = 'vanilla'
    weight_decay = 'L2'
    dropout = 'DROP'
    batchnorm = 'BN'

class Model(metaclass=ABCMeta):
    @property
    @abstractmethod
    def last_layer_idx(self):

        pass

    @abstractmethod
    def name(self): pass

    @abstractmethod
    def path(self): pass


class AlexNetCI(Model):
    last_layer_idx = 6

    def __init__(self, models_root_path, pretrained=True, create=False):
        if not os.path.exists(os.path.dirname(models_root_path)):
            raise Exception("MODEL ROOT PATH FOR ALEXNET DOES NOT EXIST: ", models_root_path)

        name = ["alexnetCI"]
        if pretrained:
            name.append("pretrained_imgnet")
        else:
            name.append("scratch")
        self.name = '_'.join(name)
        self.path = os.path.join(models_root_path,
                                 self.name + ".pth.tar")

        if not os.path.exists(self.path):
            if create:
                torch.save(alexnetCI(pretrained=pretrained), self.path)
                print("SAVED NEW ALEXNET MODEL (name=", self.name, ") to ", self.path)
            else:
                raise Exception("Not creating non-existing model: ", self.name)
        else:
            print("STARTING FROM EXISTING ALEXNET MODEL (name=", self.name, ") to ", self.path)
            print(view_saved_model(self.path))

    def name(self):
        return self.name

    def path(self):
        return self.path

def view_saved_model(path):

    print(torch.load(path))
