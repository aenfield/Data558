import os
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
import itertools

model_dir = 'imagenet'
images_dir = 'images/train'

image_dirs_train = list(os.walk(images_dir))[1:] # skip first because we don't want the parent dir
image_paths_all = list(itertools.chain.from_iterable([[os.path.join(dirinfo[0], fileinfo) for fileinfo in dirinfo[2]] for dirinfo in image_dirs_train]))
list_images = [image_path for image_path in image_paths_all if re.search('jpg|JPG', image_path)]

def create_graph():
    with gfile.FastGFile(os.path.join(model_dir,
                                      'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def extract_features(list_images):
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
            labels.append(image.split('/')[2].split('.')[1])

    return features, labels


features,labels = extract_features(list_images)

pickle.dump(features, open('features', 'wb'))
pickle.dump(labels, open('labels', 'wb'))
