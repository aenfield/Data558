from shared_routines import *

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import pickle

model_dir = 'imagenet'

def create_graph():
    with gfile.FastGFile(os.path.join(model_dir,
                                      'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

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

