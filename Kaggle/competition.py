import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime

plt.rcParams["figure.figsize"] = (25,25)

from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA

import importlib

import finalproj as fp


# fit and train convenience functions
def fit_model(model, X_train, y_train):
    output_text_with_time("Starting model fit at {}...")
    model.fit(X_train, y_train)
    output_text_with_time("Finished model fit at {}.")

def test_model_and_output_stats(model, X_test, y_actual, unique_labels, model_desc=None):
    output_text_with_time("Starting prediction at {}...")
    y_pred = model.predict(X_test)
    output_error_and_cm_for_classifier(model, y_actual, y_pred, unique_labels, "{}-cm.png".format(model_desc))
    output_text_with_time("Finished prediction at {}.")

# output, plotting
def output_text_with_time(text):
    print(text.format(datetime.now()))

def output_error_and_cm_for_classifier(fit_model, y_actual, y_pred, unique_labels, filename=None):
    print("Misclassification error: {}.".format(1-accuracy_score(y_actual, y_pred)))
    fp.plot_multiclass_confusion_matrix(confusion_matrix(y_actual, y_pred), unique_labels, show_annot=False, filename=filename)


def main():
    output_text_with_time("Loading training features at {}...")
    features_train = pd.read_csv('features_train.csv', header=None).values
    labels_train = pd.read_csv('labels_train.csv', header=None).values.ravel()
    print("Loaded {} features and {} labels at {}.".format(features_train.shape, labels_train.shape, datetime.now()))
    unique_labels = np.unique(labels_train)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(features_train, labels_train, test_size=0.1)
    print("Split sizes: {}, {}, {}, {}.".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    model_desc = "LinearSVC-C=1-squared_hinge_loss-L2_regularization-OvR-no_PCA_on_features"
    print(model_desc)
    clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr')
    fit_model(clf, X_train, y_train)
    test_model_and_output_stats(clf, X_test, y_test, unique_labels, model_desc)

    output_text_with_time("Finished at {}.")

    # features_test = pd.read_csv('features_test.csv', header=None).values
    # labels_test = pd.read_csv('labels_test.csv', header=None).values.ravel()
    # print(features_test.shape, labels_test.shape)
    #
    # clf_all = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr')
    # clf_all.fit(features_train, labels_train)
    #
    # test_pred = clf_all.predict(features_test)
    # print(test_pred.shape)
    #
    # test_results_for_output = pd.DataFrame({
    #     'Id': pd.Series(labels_test).astype(int),  # use labels as they go 1, 10, 100, etc.
    #     'Prediction': pd.Series(test_pred).apply(lambda pred: pred.split('.')[0].lstrip('0'))
    # })
    # print(test_results_for_output.shape)
    #
    # print(len(set(labels_train)), len(test_results_for_output['Prediction'].unique()))
    #
    # test_results_for_output.sort_values('Id', inplace=True)
    #
    # filename = 'pyfile_test_simple_linearsvc_from_ipynb_mirrored'
    # test_results_for_output.to_csv('{}.csv'.format(filename), index=False)
    # pickle.dump(clf_all, open('{}_model.pickle'.format(filename), 'wb'))



if __name__ == '__main__':
    main()