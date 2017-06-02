import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import datetime

plt.rcParams["figure.figsize"] = (25,25)

from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

import importlib

import finalproj as fp

section_print_delimiter = "\n---"

def fit_test_and_save_model(model_desc, model, data, unique_labels):
    """
    Do it all: fit, test, save the fit model, and return metrics.
    """
    print(section_print_delimiter)
    print("Processing '{}'.".format(model_desc))
    start_overall = output_text_with_time("Started at {}.")

    X_train, X_test, y_train, y_test = data

    start_fit = output_text_with_time("Starting model fit at {}...")
    model.fit(X_train, y_train)
    end_fit = output_text_with_time("Finished model fit at {}.")

    output_text_with_time("Starting prediction at {}...")
    y_pred = model.predict(X_test)
    output_text_with_time("Finished prediction at {}.")

    accuracy = accuracy_score(y_test, y_pred)
    print("Misclassification error: {0:.1%}.".format(1 - accuracy))
    fp.plot_multiclass_confusion_matrix(confusion_matrix(y_test, y_pred), unique_labels,
                                        show_annot=False, filename="{}-{}-cm".format(model_desc, file_datetime_now()))
    open("{}-{}-report.txt".format(model_desc, file_datetime_now()), 'w').write(classification_report(y_test, y_pred))

    output_text_with_time("Starting model save at {}...")
    pickle.dump(model, open('{}-{}-model.pickle'.format(model_desc, file_datetime_now()), 'wb'))
    output_text_with_time("Finished model save at {}.")

    end_overall = output_text_with_time("Finished at {}.")

    metrics = {'Desc': model_desc,
               'Accuracy': "{0:.1%}".format(accuracy),
               'Train time': get_formatted_time_difference(start_fit, end_fit),
               'Overall time': get_formatted_time_difference(start_overall, end_overall)
               }

    return metrics


# output, plotting
def output_text_with_time(text):
    time = datetime.datetime.now()
    print(text.format(time))
    return time

def file_datetime_now():
    return "{:%Y%m%d-%H%M%S}".format(datetime.datetime.now())

# seems a bit odd that Python doesn't support timedelta formatting better out of the box, but oh well
# modified from https://stackoverflow.com/questions/8906926/formatting-python-timedelta-objects/8907269#8907269
def strfdelta(tdelta, string_to_format):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return string_to_format.format(**d)

def get_formatted_time_difference(start_time, end_time):
    return strfdelta((end_time - start_time), "{hours:02d}:{minutes:02d}:{seconds:02d}")

def output_error_and_cm_for_classifier(fit_model, y_actual, y_pred, unique_labels, filename=None):
    print("Misclassification error: {}.".format(1-accuracy_score(y_actual, y_pred)))
    fp.plot_multiclass_confusion_matrix(confusion_matrix(y_actual, y_pred), unique_labels, show_annot=False, filename=filename)


def main():
    print(section_print_delimiter)
    output_text_with_time("Loading training features at {}...")
    features_train = pd.read_csv('features_train.csv', header=None).values
    labels_train = pd.read_csv('labels_train.csv', header=None).values.ravel()
    print("Loaded {} features and {} labels at {}.".format(features_train.shape, labels_train.shape, datetime.datetime.now()))
    unique_labels = np.unique(labels_train)

    # make it small to test
    # features_train = features_train[:1000]
    # labels_train = labels_train[:1000]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(features_train, labels_train, test_size=0.1)
    data = (X_train, X_test, y_train, y_test)
    print("Split sizes, non-PCA: {}, {}, {}, {}.".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    # use a scaler so we can scale only based on training data and then scale the validation test data using the same
    # transform; i don't think i need to reuse this for the Kaggle test data though because I don't need to worry about stuff
    # leaking through there? do I?
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_of_pca_components = 64
    pca = PCA(n_components=num_of_pca_components).fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    data_pca = (X_train_pca, X_test_pca, y_train, y_test)
    print("Split sizes, PCA: {}, {}, {}, {}.".format(X_train_pca.shape, X_test_pca.shape, y_train.shape, y_test.shape))


    # note: in model_desc use 'p' instead of a '.' (for ex, 0p01 instead of 0.01) to avoid matplotlib figure save issue with periods
    models = [("MyLogisticRegression-C=1-max_iter=100-OvR-PCA",
               OneVsRestClassifier(fp.MyLogisticRegression(max_iter=100)),
               data_pca),
              ("MyLogisticRegression-C=1-max_iter=300-OvR-PCA",
               OneVsRestClassifier(fp.MyLogisticRegression(max_iter=300)),
               data_pca),
              ("MyLogisticRegression-C=1-max_iter=100-OvO-PCA",
               OneVsOneClassifier(fp.MyLogisticRegression(max_iter=100), n_jobs=-1),
               data_pca),
              ("LinearSVC-C=1-squared_hinge_loss-L2_regularization-OvR-no_PCA_on_features",
               LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr'),
               data),
              ("LinearSVC-C=0p01-squared_hinge_loss-L2_regularization-OvR-no_PCA_on_features",
               LinearSVC(C=0.01, loss='squared_hinge', penalty='l2', multi_class='ovr'),
               data)]

    model_metrics = [fit_test_and_save_model(model_desc, model, data, unique_labels) for model_desc, model, data in models]

    model_metrics_df = pd.DataFrame(model_metrics, columns=['Desc', 'Accuracy', 'Overall time', 'Train time'])
    print(section_print_delimiter)
    print("Metrics")
    print(model_metrics_df)
    model_metrics_df.to_csv('model_metrics-{}.csv'.format(file_datetime_now()))




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