import pdb
import torch
import numpy as np

from torch.autograd import Variable


def test_model(method, model, dataset_path, target_task_head_idx, target_head=None, batch_size=200, subset='test',
               per_class_stats=False, final_layer_idx=None, task_idx=None,string_name='',class_incremental=False,
               args=None,label_order_list=None):

    dsets = torch.load(dataset_path)

    try:
        if args.class_incremental_repetition:
            dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size, shuffle=False, num_workers=4)
                            for x in ['train', 'val', 'test','fixtest']}
        else:
            dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size, shuffle=False, num_workers=4)
                            for x in ['train', 'val', 'test']}
    except:
        if args.class_incremental_repetition:
            dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size, shuffle=False, num_workers=4)
                            for x in ['train', 'val','fixtest']}
        else:
            dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size, shuffle=False, num_workers=4)
                            for x in ['train', 'val']}
        print('no test set has been found')
        subset = 'val'
    if class_incremental:
        if args.class_incremental_repetition:
            dset_classes = dsets['fixtest'].class_to_idx
        else:

            dset_classes = dsets['train'].class_to_idx
    else:
        dset_classes = dsets['train'].classes

    if target_head is not None:
        if not isinstance(target_head, list):
            target_head = [target_head]
        assert target_task_head_idx == 0, "Here head idx indicates target_headlist idx"
    if hasattr(model, 'classifier'):
        final_layer_idx = str(len(model.classifier._modules) - 1)

    model.eval()
    model = model.cuda()



    holder = type("Holder", (object,), {})()
    holder.task_imgfolders = dsets
    holder.batch_size = batch_size
    holder.model = model
    holder.dset_classes = dset_classes
    holder.heads = target_head
    holder.current_head_idx = target_task_head_idx
    holder.final_layer_idx = final_layer_idx
    holder.task_idx = task_idx

    if class_incremental:
        if args.class_incremental_repetition:
            if args.method_name=='DDGR':
                class_correct = {dset_classes[i]: 0. for i in range(len(dset_classes))}
                class_total = {dset_classes[i]: 0. for i in range(len(dset_classes))}
                this_task_class_to_idx = {dset_classes[i]: i for i in range(len(dset_classes))}
            else:
                class_correct = {dset_classes[i]:0. for i in  range(len(dset_classes))}
                class_total = {dset_classes[i]:0. for i in  range(len(dset_classes))}
        else:
            if args.method_name=='DDGR':
                class_correct = {dset_classes[i]: 0. for i in range(len(dset_classes))}
                class_total = {dset_classes[i]: 0. for i in range(len(dset_classes))}
                this_task_class_to_idx = {dset_classes[i]: i for i in range(len(dset_classes))}
            else:
                class_correct = {dset_classes[i]: 0. for i in range(len(dset_classes))}
                class_total = {dset_classes[i]: 0. for i in range(len(dset_classes))}
    else:
        class_correct = list(0. for i in range(len(dset_classes)))
        class_total = list(0. for i in range(len(dset_classes)))
    batch_count = 0

    if args.class_incremental_repetition:
        dl = dset_loaders['fixtest']
    else:
        dl = dset_loaders[subset]
    for data in dl:
        batch_count += 1
        images, labels = data
        images = images.cuda()
        images = images.squeeze()
        labels = labels.cuda()
        labels = labels.squeeze()

        if class_incremental and (args.method_name=='DDGR'):
            outputs = method.get_output(images, holder)

            _, target_head_pred = torch.max(outputs.data, 1)

            l = [this_task_class_to_idx[labels[i].item()] for i in range(len(labels))]
            ll = torch.tensor(l).reshape(labels.shape).cuda()

            c = (target_head_pred == ll).squeeze()

        else:
            outputs = method.get_output(images, holder)

            _, target_head_pred = torch.max(outputs.data, 1)

            c = (target_head_pred == labels).squeeze()

        for i in range(len(target_head_pred)):
            label = labels[i].item()
            if class_incremental:
                if label in class_total and label in class_correct:
                    class_total[label] += 1
                    class_correct[label] += c.item() if len(c.shape) == 0 else c[i].item()
            else:
                class_total[label] += 1
                class_correct[label] += c.item() if len(c.shape) == 0 else c[i].item()


        del images
        del labels
        del outputs
        del data

    if class_incremental:
        sum1 = 0.
        sum2 = 0.
        for i in range(len(dset_classes)):
            cl = dset_classes[i]
            sum1 += class_correct[cl]
            sum2 += class_total[cl]
        accuracy = sum1 * 100 / sum2
    else:
        accuracy = np.sum(class_correct) * 100 / (np.sum(class_total))

    string_name += '\tAccuracy: {:.8f}\n'.format(accuracy)
    return accuracy,string_name

