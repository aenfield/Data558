import numpy as np
import pandas as pd
import pickle

# Loads the pickled files created by images_to_features.py and saves the contents as csv files, to avoid
# issues with different versions of pickle, different processor architectures, etc.

def main():
    files = ['features_train', 'labels_train', 'features_test', 'labels_test']

    for file in files:
        print("Processing {}...".format(file))
        data_from_pickle = np.array(pickle.load(open(file, 'rb')))
        output_file_path = '{}.csv'.format(file)
        pd.DataFrame(data_from_pickle).to_csv(output_file_path, header=False, index=False)
        print("Saved {}.".format(output_file_path))

if __name__ == "__main__":
   main()
