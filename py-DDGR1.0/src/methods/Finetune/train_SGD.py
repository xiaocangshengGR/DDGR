import time
import os

import torch
from torch.autograd import Variable
from tqdm import tqdm
import src.utilities.utils as utils


def set_lr(optimizer, lr, count, decay_threshold=10, early_stop_threshold=20):

    continue_training = True

    if count > early_stop_threshold:
        continue_training = False
        print("training terminated")

    if count == decay_threshold:
        lr = lr * 0.1
        print('lr is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer, lr, continue_training


def termination_protocol(since, best_acc):

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


def train_model(model, criterion, optimizer, lr, dset_loaders, dset_sizes, use_gpu, num_epochs, exp_dir='./',
                resume='', save_models_mode=True, saving_freq=5, print_freq=100,args=None,current_label_list=None):

    if current_label_list is not None:
        this_task_class_to_idx = {current_label_list[i]: i for i in range(len(current_label_list))}
    print('dictionary length' + str(len(dset_loaders)))
    since = time.time()
    val_beat_counts = 0
    best_acc = 0.0
    mem_snapshotted = False

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        lr = checkpoint['lr']
        print("lr is ", lr)
        val_beat_counts = checkpoint['val_beat_counts']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(resume))

    if use_gpu:
        model = model.cuda()
        print("MODEL LOADED IN GPU")

    print(str(start_epoch))
    pbar = tqdm(range(start_epoch, num_epochs))
    for epoch in pbar:
        string_name = ''

        for phase in ['train', 'val']:
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            progress = ProgressMeter(
                len(dset_loaders[phase]),
                [batch_time, data_time], prefix="Epoch: [{}]".format(epoch))

            if phase == 'train':
                optimizer, lr, continue_training = set_lr(optimizer, lr, count=val_beat_counts)
                if not continue_training:
                    termination_protocol(since, best_acc)
                    return model, best_acc
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            end = time.time()
            for i, (inputs, labels) in enumerate(dset_loaders[phase]):

                data_time.update(time.time() - end)

                if args.class_incremental:
                    l = [this_task_class_to_idx[labels[i].item()] for i in range(len(labels))]
                    ll = torch.tensor(l).reshape(labels.shape)
                    labels = ll

                if 'mnist' in args.ds_name:
                    inputs = inputs.squeeze()
                if use_gpu:
                    inputs = inputs.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                model.zero_grad()

                if args.class_incremental:
                    outputs = model(inputs,current_label_list)
                else:
                    outputs = model(inputs)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels.long())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                    batch_time.update(time.time() - end)
                    end = time.time()

                if not mem_snapshotted:
                    utils.save_cuda_mem_req(exp_dir)
                    mem_snapshotted = True

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data)

                if i + 1 % print_freq == 0:
                    progress.display(i)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects.item() / dset_sizes[phase]


            string_name += '{} Loss: {:.4f} Acc: {:.4f} '.format(
                phase, epoch_loss, epoch_acc)
            if phase == 'val':
                if epoch_acc > best_acc:
                    del outputs, labels, inputs, loss, preds
                    best_acc = epoch_acc
                    if save_models_mode:
                        torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))
                        string_name += "-> new best model! "
                    else:
                        string_name += "#" * 18 + ' '
                    val_beat_counts = 0
                else:
                    val_beat_counts += 1
                    string_name += "#" * 18 + ' '
        pbar.set_description('Epoch {}/{}: '.format(epoch, num_epochs - 1) + string_name)

        if save_models_mode and epoch % saving_freq == 0:
            epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'lr': lr,
                'val_beat_counts': val_beat_counts,
                'epoch_acc': epoch_acc,
                'best_acc': best_acc,
                'arch': 'alexnet',
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch_file_name)

    termination_protocol(since, best_acc)
    return model, best_acc


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)



class AverageMeter(object):


    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
