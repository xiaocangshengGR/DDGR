import bisect
import os
import os.path
import time

from PIL import Image
import numpy as np
import copy
from itertools import accumulate

import torch
import torch.utils.data as data
from torchvision import datasets

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):

    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:

        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(dir, class_to_idx, file_list):
    images = []

    dir = os.path.expanduser(dir)
    set_files = [line.rstrip('\n') for line in open(file_list)]
    for target in sorted(os.listdir(dir)):

        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    dir_file = target + '/' + fname

                    if dir_file in set_files:
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
    return images


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderTrainVal(datasets.ImageFolder):
    def __init__(self, root, files_list, transform=None, target_transform=None,
                 loader=default_loader, classes=None, class_to_idx=None, imgs=None):

        if classes is None:
            assert class_to_idx is None
            classes, class_to_idx = find_classes(root)
        elif class_to_idx is None:
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        print("Creating Imgfolder with root: {}".format(root))
        imgs = make_dataset(root, class_to_idx, files_list) if imgs is None else imgs
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: {}\nSupported image extensions are: {}".
                                format(root, ",".join(IMG_EXTENSIONS))))
        self.root = root
        self.samples = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def get_allfigs_filepath(self):
        return self.samples
    def get_root(self):
        return self.root
    def get_classes(self):
        return self.classes
    def get_class_to_idx(self):
        return self.class_to_idx
    def get_trans(self):
        return self.transform


class ImageFolder_Subset(ImageFolderTrainVal):

    def __init__(self, dataset, indices):
        self.__dict__ = copy.deepcopy(dataset).__dict__
        self.indices = indices

    def __getitem__(self, idx):
        return super().__getitem__(self.indices[idx])

    def __len__(self):
        return len(self.indices)
    def get_allfigs_filepath(self):
        return [self.samples[x] for x in self.indices]


class ImageFolder_Subset_ClassIncremental(ImageFolder_Subset):


    def __init__(self, imgfolder_subset, target_idx):

        if not isinstance(imgfolder_subset, ImageFolder_Subset):
            print("Not a subset={}".format(imgfolder_subset))
            imagefolder_subset = random_split(imgfolder_subset, [len(imgfolder_subset)])[0]
            print("A subset={}".format(imagefolder_subset))


        imgfolder_subset = copy.deepcopy(imgfolder_subset)


        imgfolder_subset.class_to_idx = {label: idx for label, idx in imgfolder_subset.class_to_idx.items()
                                         if idx == target_idx}
        assert len(imgfolder_subset.class_to_idx) == 1
        imgfolder_subset.classes = next(iter(imgfolder_subset.class_to_idx))

        orig_samples = np.asarray(imgfolder_subset.samples)
        print(type(imgfolder_subset))
        time.sleep(5)
        subset_samples = orig_samples[imgfolder_subset.indices.numpy()]

        print("SUBSETTING 1 CLASS FROM DSET WITH SIZE: ", subset_samples.shape[0])

        label_idxs = np.where(subset_samples[:, 1] == str(target_idx))[0]  # indices row
        print("#SAMPLES WITH LABEL {}: {}".format(target_idx, label_idxs.shape[0]))

        final_indices = imgfolder_subset.indices[label_idxs]


        is_all_same_label = str(target_idx) == orig_samples[final_indices, 1]
        assert np.all(is_all_same_label)

        super().__init__(imgfolder_subset, final_indices)


class ImageFolder_Subset_PathRetriever(ImageFolder_Subset):

    def __init__(self, imagefolder_subset):
        if not isinstance(imagefolder_subset, ImageFolder_Subset):
            print("Transforming into Subset Wrapper={}".format(imagefolder_subset))
            imagefolder_subset = random_split(imagefolder_subset, [len(imagefolder_subset)])[0]
        super().__init__(imagefolder_subset, imagefolder_subset.indices)

    def __getitem__(self, index):

        original_tuple = super(ImageFolder_Subset_PathRetriever, self).__getitem__(index)

        path = self.samples[self.indices[index]][0]

        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path


class ImagePathlist(data.Dataset):


    def __init__(self, imlist, targetlist=None, root='', transform=None, loader=default_loader):
        self.imlist = imlist
        self.targetlist = targetlist
        self.root = root
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]

        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        if self.targetlist is not None:
            target = self.targetlist[index]
            return img, target
        else:
            return img

    def __len__(self):
        return len(self.imlist)


def random_split(dataset, lengths):

    assert sum(lengths) == len(dataset)
    indices = torch.randperm(sum(lengths))
    return [ImageFolder_Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(accumulate(lengths), lengths)]


class ConcatDatasetDynamicLabels(torch.utils.data.ConcatDataset):


    def __init__(self, datasets, classes_len):

        super(ConcatDatasetDynamicLabels, self).__init__(datasets)
        self.cumulative_classes_len = list(accumulate(classes_len))

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
            img, label = self.datasets[dataset_idx][sample_idx]
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            img, label = self.datasets[dataset_idx][sample_idx]
            label = label + self.cumulative_classes_len[dataset_idx - 1]  # Shift Labels
        return img, label
