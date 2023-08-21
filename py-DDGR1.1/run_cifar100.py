
import os

statue_out = []

cmd = "CUDA_VISIBLE_DEVICES=0 python main.py --class_incremental --method_name DDGR --model_name alexnetCI_pretrained " \
                "--ds_name cifar100CI --batch_size 256 --num_epochs 100 --drop_margin 0.5 --max_attempts_per_task 1 " \
                "--test --test_set test --test_overwrite_mode --max_task_count 5 --test_max_task_count 5 --CI_task_count 10 " \
                "--classifier_scale 1.0 " \
                "--attention_resolutions 32,16,8 " \
                "--class_cond True " \
                "--diffusion_steps 4000 " \
                "--dropout 0.3 " \
                "--image_size 64 " \
                "--learn_sigma True " \
                "--noise_schedule cosine " \
                "--num_channels 128 " \
                "--num_head_channels 64 " \
                "--num_res_blocks 3 " \
                "--resblock_updown True " \
                "--use_new_attention_order True " \
                "--use_fp16 True " \
                "--log_interval 50 " \
                "--use_scale_shift_norm True " \
                "--lr_anneal_steps 15000 --save_interval 500 " \
                "--diffusion_lr 1e-4 " \
                "--diffusion_batch_size 64 " \
                "--classifier_batch_size 32 " \
                "--classifier_depth 4 " \
                "--num_samples 20 --timestep_respacing 250 " \
                "--DDGR_generator_factor 0.25 "
os.system(cmd)
cmd = "CUDA_VISIBLE_DEVICES=0 python main.py --class_incremental --method_name DDGR --model_name alexnetCI_pretrained " \
                "--ds_name cifar100CI --batch_size 256 --num_epochs 100 --drop_margin 0.5 --max_attempts_per_task 1 " \
                "--test --test_set test --test_overwrite_mode --max_task_count 5 --test_max_task_count 5 --CI_task_count 5 " \
                "--classifier_scale 1.0 " \
                "--attention_resolutions 32,16,8 " \
                "--class_cond True " \
                "--diffusion_steps 4000 " \
                "--dropout 0.3 " \
                "--image_size 64 " \
                "--learn_sigma True " \
                "--noise_schedule cosine " \
                "--num_channels 128 " \
                "--num_head_channels 64 " \
                "--num_res_blocks 3 " \
                "--resblock_updown True " \
                "--use_new_attention_order True " \
                "--use_fp16 True " \
                "--log_interval 50 " \
                "--use_scale_shift_norm True " \
                "--lr_anneal_steps 15000 --save_interval 500 " \
                "--diffusion_lr 1e-4 " \
                "--diffusion_batch_size 64 " \
                "--classifier_batch_size 32 " \
                "--classifier_depth 4 " \
                "--num_samples 20 --timestep_respacing 250 " \
                "--DDGR_generator_factor 0.25 "
os.system(cmd)



