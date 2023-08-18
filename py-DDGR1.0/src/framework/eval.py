import sys
import torch
import os
import traceback
import time

import src.data.dataset as datasets
import src.methods.method as methods
import src.utilities.utils as utils


def main(args, manager, ds_paths, model_paths):

    args.test_max_task_count = manager.dataset.task_count if args.test_max_task_count is None else args.test_max_task_count
    ds_paths = ds_paths[0] if len(ds_paths) == 1 and isinstance(ds_paths[0], list) else ds_paths  # Joint

    args.out_path = utils.get_test_results_path(args.test_results_root_path, manager.dataset,
                                                method_name=manager.method.eval_name,
                                                model_name=manager.base_model.name,
                                                gridsearch_name=args.gridsearch_name,
                                                exp_name=args.exp_name,
                                                subset=args.test_set,
                                                create=True)

    args.records_path = utils.get_records_path(args.test_results_root_path, manager.dataset,
                                                method_name=manager.method.eval_name,
                                                model_name=manager.base_model.name,
                                                gridsearch_name=args.gridsearch_name,
                                                exp_name=args.exp_name,
                                                subset=args.test_set,
                                                create=True)

    max_out_filepath = os.path.join(args.out_path, utils.get_perf_output_filename(manager.method.eval_name,
                                                                                  manager.dataset.task_count - 1))

    if not args.debug and not args.test_overwrite_mode and os.path.exists(max_out_filepath):
        print("[OVERWRITE=False] SKIPPING EVAL, Already exists: ", max_out_filepath)
        exit(0)

    args.task_lengths = datasets.get_nc_per_task(manager.dataset)

    if hasattr(manager.method, 'eval_model_preprocessing'):
        model_paths = manager.method.eval_model_preprocessing(args)

    print("\nTESTING ON DATASETS:")
    print('\n'.join(ds_paths))

    print("\nTESTING ON MODELS:")
    print('\n'.join(model_paths))

    print("Testing on ", len(ds_paths), " task datasets")

    if manager.method.name == methods.Joint.name:
        args.model_path = model_paths[0]
        eval_single_model_all_tasks(args, manager, ds_paths, )
    else:
        eval_all_models_all_tasks(args, manager, ds_paths, model_paths)

    utils.print_stats()
    print("FINISHED testing of: ", args.exp_name)


def eval_single_model_all_tasks(args, manager, ds_paths):

    args.task_counter = manager.dataset.task_count
    args.task_name = "TEST ALL"
    joint_ds_path = manager.method.grid_datafetch(args, manager.dataset)
    joint_dataloader = None
    joint_class_to_fc_idx = None
    if not isinstance(joint_ds_path, list):
        joint_dataset = torch.load(joint_ds_path)
        joint_classes = joint_dataset['train'].classes
        del joint_dataset
        joint_class_to_fc_idx = {class_name: idx for idx, class_name in enumerate(joint_classes)}
    else:
        joint_dataloader, _, dset_classes = manager.method.compose_dataset(joint_ds_path, args.batch_size)
        joint_classes = sum(dset_classes['train'], [])

    args.tasks_idxes = []
    tasks_classes = []
    tasks_output_count = 0
    for dataset_idx, dataset_path in enumerate(ds_paths):
        task_dataset = torch.load(dataset_path)
        task_classes = task_dataset['train'].classes
        if joint_class_to_fc_idx is not None:
            task_idxes = [joint_class_to_fc_idx[task_class] for task_class in task_classes
                          if task_class in joint_class_to_fc_idx]
        elif joint_dataloader is not None:
            if dataset_idx == 0:
                task_idxes = [idx for idx in range(len(task_classes))]
            else:
                task_idxes = [idx + joint_dataloader['train'].dataset.cumulative_classes_len[dataset_idx - 1]
                              for idx in range(len(task_classes))]
        else:
            raise Exception()
        args.tasks_idxes.append(sorted(task_idxes))
        tasks_classes.append(task_classes)
        tasks_output_count += len(task_classes)

    print("{} classes in joint set, {} classes in separate sets".format(len(joint_classes), tasks_output_count))
    assert len(joint_classes) == tasks_output_count

    acc_all = {}
    method_performances = {manager.method.eval_name: {}}
    out_filepath = os.path.join(args.out_path, utils.get_perf_output_filename(manager.method.eval_name, None,
                                                                              joint_full_batch=True))
    try:
        for dataset_index in range(args.test_starting_task_count - 1, args.test_max_task_count):
            args.dataset_index = dataset_index
            args.dataset_path = ds_paths[dataset_index]

            acc = manager.method.inference_eval(args, manager)

            acc_all[dataset_index] = acc
            if 'seq_res' not in method_performances[manager.method.eval_name]:
                method_performances[manager.method.eval_name]['seq_res'] = []
            method_performances[manager.method.eval_name]['seq_res'].append(acc)

        if not args.debug:
            torch.save(method_performances, out_filepath)
            print("Saved results to: ", out_filepath)

        print("FINAL RESULTS: ", method_performances[manager.method.eval_name]['seq_res'])

    except Exception as e:
        print("TESTING ERROR: ", e)
        print("No results saved...")
        traceback.print_exc()


def eval_all_models_all_tasks(args, manager, ds_paths, model_paths):

    acc_all = {}
    forgetting_all = {}
    total_string = ''
    start_time = time.time()

    for dataset_index in range(args.test_starting_task_count - 1, args.test_max_task_count):
        method_performances = {manager.method.eval_name: {}}
        out_filepath = os.path.join(args.out_path,
                                    utils.get_perf_output_filename(manager.method.eval_name, dataset_index))
        args.eval_dset_idx = dataset_index
        if not args.test_overwrite_mode and not args.debug:
            if os.path.exists(out_filepath):
                print("EVAL already done, can only rerun in overwrite mode")
                break

        try:
            string_name = 'Testing_task: ' + str(dataset_index) + '\t'
            seq_acc, seq_forgetting, seq_head_acc,rstr = eval_task_steps_accuracy(args, manager, ds_paths, model_paths,string_name)
            total_string += rstr + '\n'

            if len(seq_acc[dataset_index]) == 0:
                msg = "SKIPPING SAVING: acc empty: ", seq_acc[dataset_index]
                print(msg)
                raise Exception(msg)

            acc_all[dataset_index] = seq_acc[dataset_index]
            forgetting_all[dataset_index] = seq_forgetting[dataset_index]
            method_performances[manager.method.eval_name]['seq_res'] = seq_acc
            method_performances[manager.method.eval_name]['seq_forgetting'] = seq_forgetting
            method_performances[manager.method.eval_name]['seq_head_acc'] = seq_head_acc

            if not args.debug:
                torch.save(method_performances, out_filepath)
                if not args.class_incremental:
                    print("Saved results to: ", out_filepath)

            if args.class_incremental_repetition:
                break

        except Exception as e:
            print("TESTING ERROR: ", e)
            print("No results saved...")
            traceback.print_exc()
            break
    if args.class_incremental:
        total_string = ""
        for model_idx in range(args.test_starting_task_count - 1, args.test_max_task_count):
            sum_model = 0.
            for_model = 0.
            for data_idx in range(model_idx+1):
                sum_model += acc_all[data_idx][model_idx-data_idx]
                if data_idx < model_idx:
                    for_model += forgetting_all[data_idx][model_idx-data_idx-1]
            s_name = 'Testing_model: ' + str(model_idx) + '\tAccuracy: {:.8f}\tforgetting: {:.8f}\n'.format(sum_model/(model_idx+1),
                                                                                                            for_model/(model_idx+1))
            print(s_name)
            total_string += s_name
    elapsed_time = time.time() - start_time
    if args.test_overwrite_mode:
        total_string += 'TOTAL_EVAL_time: {}'.format(elapsed_time)
        file = open(os.path.join(args.records_path,'eval_records.txt'),'w')
        file.write(total_string)
        file.close()
    utils.print_timing(elapsed_time, title="TOTAL EVAL")


class EvalMetrics(object):
    def __init__(self):
        self.seq_acc = []
        self.seq_forgetting = []
        self.seq_head_acc = []


def eval_task_steps_accuracy(args, manager, ds_paths, model_paths, string_name):

    if not args.class_incremental:
        print("TESTING ON TASK ", args.eval_dset_idx + 1)
    seq_acc = {args.eval_dset_idx: []}
    seq_forgetting = {args.eval_dset_idx: []}
    head_accuracy = None
    seq_head_acc = []

    args.dset_path = ds_paths[args.eval_dset_idx]
    args.head_paths = model_paths[args.eval_dset_idx]

    string_name += 'Testing_model:\t'
    return_string = ''
    if args.debug:
        print("Testing Dataset = ", args.dset_path)

    for trained_model_idx in range(args.eval_dset_idx, len(ds_paths)):

        args.trained_model_idx = trained_model_idx
        args.eval_model_path = model_paths[trained_model_idx]
        if args.debug:
            print("Testing on model = ", args.eval_model_path)

        try:
            ssss = string_name + str(trained_model_idx)
            accuracy,ss = manager.method.inference_eval(args, manager,ssss)

            seq_acc[args.eval_dset_idx].append(accuracy)
            if trained_model_idx > args.eval_dset_idx:
                first_task_acc = seq_acc[args.eval_dset_idx][0]
                seq_forgetting[args.eval_dset_idx].append(first_task_acc - accuracy)
            if head_accuracy is not None:
                seq_head_acc.append(head_accuracy)

            if not args.class_incremental:
                print(ss)
                return_string += ss
            else:
                print(ss)
                return_string += ss

        except Exception:
            print("ERROR in Testing model, trained until TASK ", str(trained_model_idx + 1))
            print("Aborting testing on further models")
            traceback.print_exc(5)
            break
    return seq_acc, seq_forgetting, seq_head_acc,return_string

