
import traceback
import os
import time
import torch
import copy
import sys
import operator
import src.utilities.utils as utils
import src.methods.method as methods
import src.framework.lr_grid_train as finetune_single_task


class HyperparameterFramework(object):

    def __init__(self, method):
        self.hyperparams = method.hyperparams
        self.hyperparams_backup = copy.deepcopy(self.hyperparams)
        self.init_hyperparams = copy.deepcopy(self.hyperparams)
        self.hyperparam_idx = 0
        self.attempts = 0

    def _restore_state(self, state):
        print("Restoring to state = {}".format(state))
        try:
            for hkey in self.hyperparams.keys():
                self.hyperparams[hkey] = state['hyperparams'][hkey]
                self.hyperparams_backup[hkey] = state['hyperparams_backup'][hkey]
            self.hyperparam_idx = state['hyperparam_idx']
            self.attempts = state['attempts']
        except Exception as e:
            print(e)
            raise AttributeError("Attributes of inner state have changed: src:{} != target:{}".format(
                list(self.hyperparams.keys()), list(state.keys())))
        self._print_status("Restored Framework")

    def _get_state(self):
        return {"hyperparams": self.hyperparams,
                "hyperparams_backup": self.hyperparams_backup,
                "hyperparam_idx": self.hyperparam_idx,
                "attempts": self.attempts, }

    def _print_status(self, title="Framework Status"):
        print("-" * 40, "\n", title)
        for hkey in self.hyperparams.keys():
            print('hyperparam {}={} (backup={})'.format(
                hkey, self.hyperparams[hkey], self.hyperparams_backup[hkey]))
        print("hyperparam_idx={}".format(self.hyperparam_idx))
        print("attempts={}".format(self.attempts))
        print("-" * 40)

    def _save_chkpt(self, args, manager, threshold, task_lr_acc):
        hyperparams = {
            'acc_threshold': threshold, 'val_acc': task_lr_acc,
            'args': vars(args), 'manager': vars(manager), 'state': self._get_state()
        }
        print("Saving hyperparams: {}".format(hyperparams))
        manager.save_hyperparams(manager.heuristic_exp_dir, hyperparams)

    @staticmethod
    def maximalPlasticitySearch(args, manager):
        start_time = time.time()
        finetune_lr, finetune_acc = finetune_single_task.lr_grid_single_task(args, manager,
                                                                             save_models_mode=args.save_models_mode)
        args.phase1_elapsed_time = time.time() - start_time
        utils.print_timing(args.phase1_elapsed_time, "PHASE 1 FT GRID")
        return finetune_lr, finetune_acc


    def stabilityDecay(self, args, manager, finetune_lr, finetune_acc,finetune_flag = False):

        args.lr = finetune_lr
        manager.heuristic_exp_dir = os.path.join(
            manager.parent_exp_dir, 'task_' + str(args.task_counter), 'TASK_TRAINING')
        if hasattr(manager.method, 'train_init'):
            manager.method.train_init(args, manager)

        chkpt_loaded = self.load_chkpt(manager)
        if not chkpt_loaded:
            self.hyperparamDecay(args,manager,back_decay=1/(args.decaying_factor))
            self.attempts = 0
            self.hyperparams_backup = copy.deepcopy(self.hyperparams)
        if self.check_succes(manager):
            manager.best_model_path = os.path.join(manager.heuristic_exp_dir, 'best_model.pth.tar')
            return

        args.presteps_elapsed_time = 0
        if hasattr(manager.method, 'prestep'):
            manager.method.prestep(args, manager)

        max_attempts = args.max_attempts_per_task
        converged = False
        dd = None
        while not converged and self.attempts < max_attempts:
            print(" => ATTEMPT {}/{}: Hyperparams {}".format(self.attempts, max_attempts - 1, self.hyperparams))
            start_time = time.time()
            try:
                if args.method_name == 'IF' and args.class_incremental_repetition:
                    model, task_lr_acc = manager.method.train(args, manager, self.hyperparams, dd)
                else:
                    model, task_lr_acc = manager.method.train(args, manager, self.hyperparams)
            except:
                traceback.print_exc()
                sys.exit(1)

            if not finetune_flag:
                finetune_acc = task_lr_acc
            threshold = finetune_acc * args.inv_drop_margin
            print(finetune_acc)
            print(args.inv_drop_margin)

            if task_lr_acc >= threshold:
                print('CONVERGED, (acc = ', task_lr_acc, ") >= (threshold = ", threshold, ")")
                converged = True
                args.convergence_iteration_elapsed_time = time.time() - start_time
                utils.print_timing(args.convergence_iteration_elapsed_time, "PHASE 2 CONVERGED FINAL IT")


            else:
                print('DECAY HYPERPARAMS, (acc = ', task_lr_acc, ") < (threshold = ", threshold, ")")
                self.hyperparamDecay(args, manager)
                self.attempts += 1

                if self.attempts < max_attempts:
                    print('CLEANUP of previous model')
                    utils.rm_dir(manager.heuristic_exp_dir)
                else:
                    print("RETAINING LAST ATTEMPT MODEL")
                    converged = True


            self._save_chkpt(args, manager, threshold, task_lr_acc)
            self._print_status()


        manager.method.hyperparams = self.hyperparams
        manager.best_model_path = os.path.join(manager.heuristic_exp_dir, 'best_model.pth.tar')
        manager.create_success_token(manager.heuristic_exp_dir)

    def check_succes(self, manager):

        if os.path.exists(manager.get_success_token_path(manager.heuristic_exp_dir)):

            return True
        return False

    def load_chkpt(self, manager):
        utils.create_dir(manager.heuristic_exp_dir)
        hyperparams_path = os.path.join(manager.heuristic_exp_dir, utils.get_hyperparams_output_filename())
        try:
            print("Initiating framework chkpt:{}".format(hyperparams_path))
            chkpt = torch.load(hyperparams_path)
        except:
            print("CHECKPOINT LOAD FAILED: No state to restore, starting from scratch.")
            return False

        self._restore_state(chkpt['state'])
        print("SUCCESSFUL loading framework chkpt:{}".format(hyperparams_path))
        return True

    def hyperparamDecay(self, args, manager,back_decay = None):

        op = manager.method.decay_operator if hasattr(manager.method, 'decay_operator') else operator.mul


        factor = args.decaying_factor
        if not back_decay ==None:
            factor = back_decay
        if len(self.hyperparams) == 1:
            hkey, hval = list(self.hyperparams.items())[0]
            before = hval
            self.hyperparams[hkey] = op(self.hyperparams[hkey], factor)
            print("Decayed {} -> {}".format(before, self.hyperparams[hkey]))

        else:
            if self.hyperparam_idx == len(self.hyperparams):
                self.hyperparam_idx = 0
                for hkey, hval in self.hyperparams_backup.items():
                    self.hyperparams[hkey] = op(hval, factor)
                before = copy.deepcopy(self.hyperparams_backup)
                self.hyperparams_backup = copy.deepcopy(self.hyperparams)
                print("DECAYING ALL HYPERPARAMS: {} -> {}".format(before, self.hyperparams))

            else:
                before = copy.deepcopy(self.hyperparams)
                hlist = list(self.hyperparams.items())
                hkey = hlist[self.hyperparam_idx][0]
                self.hyperparams[hkey] = op(self.hyperparams_backup[hkey], factor)
                other_keys = [hlist[i][0] for i in range(len(self.hyperparams)) if i != self.hyperparam_idx]
                for other_key in other_keys:
                    self.hyperparams[other_key] = self.hyperparams_backup[other_key]
                self.hyperparam_idx += 1
                print("Decayed 1 hyperparam: {} -> {}".format(before, self.hyperparams))


def framework_single_task(args, manager):

    if args.task_counter == 1 and not args.train_first_task and not args.wrap_first_task_model:
        print("USING SI AS MODEL FOR FIRST TASK: ", manager.previous_task_model_path)
        return


    skip_to_post = args.wrap_first_task_model and args.task_counter == 1
    hf = HyperparameterFramework(manager.method)

    if args.save_models_FT_heuristic:
        args.save_models_mode = 'all'
    else:
        args.save_models_mode = 'keep_none'
    args.phase1_elapsed_time = 0
    args.presteps_elapsed_time = 0
    args.convergence_iteration_elapsed_time = 0
    args.postprocess_time = 0
    print("HEURISTIC BASED METHOD: Task ", args.task_name)

    if args.task_counter > 1:
        prev_task_name = manager.dataset.get_taskname(args.task_counter - 1)
        args.previous_task_dataset_path = manager.dataset.get_task_dataset_path(task_name=prev_task_name,
                                                                                rnd_transform=False)

        manager.reg_sets = [manager.dataset.get_task_dataset_path(task_name=prev_task_name, rnd_transform=False)]
        print('reg_sets=', manager.reg_sets)

    args.classifier_heads_starting_idx = manager.base_model.last_layer_idx
    print("classifier_heads_starting_idx = ", args.classifier_heads_starting_idx)

    if not skip_to_post:

        print("\nPHASE 1 (TASK {})".format(args.task_counter))
        if args.ds_name == 'core50CIREP':
            ft_acc = 0.8
            ft_lr = 1e-4
        elif args.ds_name == 'imagenet1000CI':
            ft_acc = 0.7
            ft_lr = 1e-2
        elif args.ds_name == 'cifar100CI':
            ft_acc = 0.4
            ft_lr = 1e-3
        else:
            ft_lr, ft_acc = hf.maximalPlasticitySearch(args, manager)
        finetune_flag = True
        print("\nPHASE 2 (TASK {})".format(args.task_counter))
        print("*" * 20, " FT LR ", ft_lr, "*" * 20)
        hf.stabilityDecay(args, manager, ft_lr, ft_acc,finetune_flag)

    if hasattr(manager.method, 'poststep'):
        manager.method.poststep(args, manager)

    if hasattr(manager.method, 'init_next_task'):
        manager.method.init_next_task(manager)
    else:
        manager.previous_task_model_path = manager.best_model_path

    print('phase1_elapsed_time={}, '
          'presteps_elapsed_time={}, '
          'convergence_iteration_elapsed_time={}, '
          'postprocess_time={}'.format(args.phase1_elapsed_time,
                                       args.presteps_elapsed_time,
                                       args.convergence_iteration_elapsed_time,
                                       args.postprocess_time))