import math
import torch
import os
import time
from torch.autograd import Variable

import src.utilities.utils as utils



def train_model(args, model, criterion, optimizer, lr, dset_loaders, dset_sizes, use_cuda, num_epochs,
                task_counter,exp_dir='./',
                resume='', saving_freq=5,device=None,combine_label_list=None,gen_dset_loaders=None):

    this_task_class_to_idx = {combine_label_list[i]: i for i in range(len(combine_label_list))}
    print('dictoinary length' + str(len(dset_loaders)))
    since = time.time()
    mem_snapshotted = False
    val_beat_counts = 0
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

        model.load_state_dict(checkpoint['state_dict'])
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = checkpoint['lr']
        print("lr is ", lr)
        val_beat_counts = checkpoint['val_beat_counts']

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(resume))

    print(str(start_epoch))
    print("lr is", lr)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print('task_counter: '+str(task_counter))

        for phase in ['train', 'val']:

            if phase == 'train':
                optimizer, lr, continue_training = set_lr(optimizer, lr, count=val_beat_counts)
                if not continue_training:
                    traminate_protocol(since, best_acc)
                    return model, best_acc
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0
            running_counter = 0
            if gen_dset_loaders is not None:
                ziploaders = enumerate(zip(dset_loaders[phase],gen_dset_loaders[phase]))
            else:
                ziploaders = enumerate(dset_loaders[phase])

            for _,data in ziploaders:

                if gen_dset_loaders is not None:
                    inputs, labels = data[0]
                    gen_inputs, gen_labels = data[1]
                    inputs = torch.cat((inputs,gen_inputs))
                    labels = torch.cat((labels,gen_labels))
                else:
                    inputs, labels = data
                if 'mnist' in args.ds_name:
                    inputs = inputs.squeeze()
                if args.class_incremental or args.class_incremental_repetition:
                    l = [this_task_class_to_idx[labels[i].item()] for i in range(len(labels))]
                    ll = torch.tensor(l).reshape(labels.shape)
                    labels = ll

                if use_cuda:
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                running_counter+=inputs.shape[0]
                optimizer.zero_grad()

                if args.class_incremental or args.class_incremental_repetition:
                    logits = model(inputs,combine_label_list)
                else:
                    logits = model(inputs)
                _, preds = torch.max(logits.data, 1)

                loss = criterion(logits, labels)


                if phase == 'train':

                    loss.backward()

                    optimizer.step()

                if not mem_snapshotted:
                    utils.save_cuda_mem_req(exp_dir)
                    mem_snapshotted = True

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / running_counter
            epoch_acc = running_corrects / running_counter

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if epoch_loss > 1e4 or math.isnan(epoch_loss):
                return model, best_acc, None

            if phase == 'val':
                if epoch_acc > best_acc:
                    del logits, labels, inputs, loss, preds
                    best_acc = epoch_acc
                    torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))
                    val_beat_counts = 0
                else:
                    val_beat_counts += 1
        if epoch % saving_freq == 0:

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
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model, best_acc

def set_lr(optimizer, lr, count):

    continue_training = True
    if count > 10:
        continue_training = False
        print("training terminated")
    if count == 5:
        lr = lr * 0.1
        print('lr is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer, lr, continue_training


def traminate_protocol(since, best_acc):
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def update_input(self, input, output):
    self.input = input[0].data
    self.output = output




