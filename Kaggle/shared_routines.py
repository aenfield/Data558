import os
import re
import itertools

images_train_dir = 'images/train'
images_test_dir = 'images/test'

def get_image_list(which="train"):
    image_paths_all = []
    if which == "train": # train, many directories
        image_dirs_train = list(os.walk(images_train_dir))[1:]  # skip first because we don't want the parent dir
        image_paths_all = list(itertools.chain.from_iterable(
            [[os.path.join(dirinfo[0], fileinfo) for fileinfo in dirinfo[2]] for dirinfo in image_dirs_train]))
    else: # test, single directory
        image_paths_all = [os.path.join(images_test_dir, file_path) for file_path in os.listdir(images_test_dir)]

    list_images = [image_path for image_path in image_paths_all if re.search('jpg|JPG', image_path)]
    return list_images

def get_label(image_path, which="train"):
    if which == "train":
        return image_path.split('/')[2]  # - we'll just output w/ the ID too to have it later - we used to use the following on the end: .split('.')[1]
    else: # test, just numbers
        return image_path.split('/')[2].split('.')[0]  # here we'll output just the number

