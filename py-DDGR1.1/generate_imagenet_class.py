
import os
import src.utilities.utils as utils
config = utils.get_parsed_config()
src_root_dir = os.path.join(utils.read_from_config(config, 'ds_root_path'), 'ImageNet', 'imagenet_resized_64-1000')
class_name_train_list = os.listdir(os.path.join(src_root_dir,'train'))
class_name_val_list = os.listdir(os.path.join(src_root_dir,'val'))
class_name_val_list.sort()
class_name_train_list.sort()
print(os.listdir(os.path.join(src_root_dir,'train')))
flag = True
if len(class_name_val_list) == len(class_name_train_list):
    for i in range(len(class_name_val_list)):
        if class_name_val_list[i] != class_name_train_list[i]:
            flag = False
            break
else:
    flag = False
if not flag:
    print('train class is not equal to val class.')
else:
    file = open(os.path.join(os.path.dirname(utils.read_from_config(config, 'ds_root_path')),'imgnet_classes.txt'),'w')

    file.writelines([line+'\n' for line in class_name_train_list])
    file.close()
    print('imgnet_classes.txt is created in {}'.format(os.path.dirname(utils.read_from_config(config, 'ds_root_path'))))