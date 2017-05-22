import os
import re

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import pickle
import itertools

model_dir = 'imagenet'
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


def create_graph():
    with gfile.FastGFile(os.path.join(model_dir,
                                      'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def get_label(image_path, which="train"):
    if which == "train":
        return image_path.split('/')[2]  # - we'll just output w/ the ID too to have it later - we used to use the following on the end: .split('.')[1]
    else: # test, just numbers
        return image_path.split('/')[2].split('.')[0]  # here we'll output just the number


def extract_features(list_images, which="train"):
    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    labels = []

    create_graph()

    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(list_images):
            if (ind%100 == 0):
                print('Processing %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind,:] = np.squeeze(predictions)
            labels.append(get_label(image, which))

    return features, labels


def main():
    which = "train"
    list_images = get_image_list(which)
    features, labels = extract_features(list_images, which)

    pickle.dump(features, open('features_{}'.format(which), 'wb'))
    pickle.dump(labels, open('labels_{}'.format(which), 'wb'))

if __name__ == "__main__":
   main()

