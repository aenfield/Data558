import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import datetime
figsize = (25,23)
plt.rcParams["figure.figsize"] = figsize

from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import GridSearchCV

import finalproj as fp

section_print_delimiter = "\n---"


def fit_test_and_save_model(model_desc, model, data, unique_labels):
    """
    Do it all: fit, test, save the fit model, and return metrics.
    """
    print(section_print_delimiter)
    print("Processing '{}'.".format(model_desc))
    start_overall = output_text_with_time("Started at {}.")
    model_notes = ""

    X_train, X_test, y_train, y_test = data

    start_fit = output_text_with_time("Starting model fit at {}...")
    model.fit(X_train, y_train)
    end_fit = output_text_with_time("Finished model fit at {}.")

    # for base estimators we just use them as is, but to use the best estimator from a GridSearchCV instance
    # we need to pull it out; we also want to log the characteristics of the estimator
    if isinstance(model, GridSearchCV):
        # we overwrite the GridSearchCV instance now that we're done with it (ok?)
        # this lets us avoid changing the pickle code below, but means we don't get back the whole CV instance; we
        # just get the best model, which I think I'll go with for now
        model = model.best_estimator_

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Misclassification error: {0:.1%}.".format(1 - accuracy))
    fp.plot_multiclass_confusion_matrix(confusion_matrix(y_test, y_pred), unique_labels,
                                        show_annot=False, filename="{}-{}-cm.png".format(model_desc, file_datetime_now()))
    open("{}-{}-report.txt".format(model_desc, file_datetime_now()), 'w').write(classification_report(y_test, y_pred))
    # output description of the classifier as a text file, save it too for the spreadsheet
    estimator_desc = str(model)
    open("{}-{}-estimator_desc.txt".format(model_desc, file_datetime_now()), 'w').write(estimator_desc)
    model_notes += estimator_desc

    model_filename = '{}-{}-model.pickle'.format(model_desc, file_datetime_now())
    print("Pickling {}.".format(model_filename))
    pickle.dump(model, open(model_filename, 'wb'))

    end_overall = output_text_with_time("Finished at {}.")

    metrics = {'Desc': model_desc,
               'Accuracy': "{0:.1%}".format(accuracy),
               'Train time': get_formatted_time_difference(start_fit, end_fit),
               'Overall time': get_formatted_time_difference(start_overall, end_overall),
               'Notes': model_notes
               }

    return metrics

def get_pca_data_and_save_transformer(X_train, X_test, y_train, y_test, num_components):
    pca = PCA(n_components=num_components).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    pickle.dump(pca, open('pca_transform_{}_components-{}.pickle'.format(num_components, file_datetime_now()), 'wb'))

    print("Split sizes, PCA {}: {}, {}, {}, {}.".format(num_components, X_train_pca.shape, X_test_pca.shape, y_train.shape, y_test.shape))
    data_pca = (X_train_pca, X_test_pca, y_train, y_test)
    return data_pca


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


def train_models():
    output_text_with_time("Loading training features at {}...")
    features_train = pd.read_csv('features_train.csv', header=None).values
    labels_train = pd.read_csv('labels_train.csv', header=None).values.ravel()
    print("Loaded {} features and {} labels at {}.".format(features_train.shape, labels_train.shape, datetime.datetime.now()))
    unique_labels = np.unique(labels_train)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(features_train, labels_train, test_size=0.1)
    data = (X_train, X_test, y_train, y_test)
    print("Split sizes, non-PCA: {}, {}, {}, {}.".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    # use a scaler so we can scale only based on training data and then scale the validation test data using the same
    # transform; i don't think i need to reuse this for the Kaggle test data though because I don't need to worry about stuff
    # leaking through there? do I?
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    data_pca_64 = get_pca_data_and_save_transformer(X_train_scaled, X_test_scaled, y_train, y_test, num_components=64)
    data_pca_256 = get_pca_data_and_save_transformer(X_train_scaled, X_test_scaled, y_train, y_test, num_components=256)

    # note: in model_desc use 'p' instead of a '.' (for ex, 0p01 instead of 0.01) to avoid matplotlib figure save issue with periods
    # Some of these are commented out as the last thing I did was generate two voting classifiers using a subset
    # This list is the upshot of a 100+ runs with different algorithms, data, and parameters
    models = [("ETC-md=None-mf=75-mss=2-210_est-no_PCA", ExtraTreesClassifier(max_depth=None, max_features=75, min_samples_split=2, n_estimators=210, n_jobs=-1), data),
              #("LogisticRegression-C=1-L2_regularization-OvR-no_PCA", LogisticRegression(fit_intercept=False), data),
              ("LogisticRegression-C=1-L2_regularization-multinomial-no_PCA", LogisticRegression(fit_intercept=False, multi_class='multinomial', solver='newton-cg'), data),
              ("RF-md=None-mf=60-mss=2-150_est-no_PCA", RandomForestClassifier(max_depth=None, max_features=60, min_samples_split=2, n_estimators=150, n_jobs=-1), data),
              ("LinearSVC-C=1-squared_hinge_loss-L2_regularization-CS-no_PCA", LinearSVC(fit_intercept=False, multi_class='crammer_singer'), data),
              ("LinearSVC-C=1-squared_hinge_loss-L2_regularization-OvR-no_PCA", LinearSVC(fit_intercept=False), data),
              #("LinearSVC-C=0p01-squared_hinge_loss-L2_regularization-OvR-no_PCA", LinearSVC(C=0.01), data),
              ("ETC-md=None-mf=30-mss=2-210_est-PCA_256", ExtraTreesClassifier(max_depth=None, max_features=30, min_samples_split=2, n_estimators=210, n_jobs=-1), data_pca_256),
              #("MyLogisticRegression-C=1-max_iter=300-OvR-PCA_256", OneVsRestClassifier(fp.MyLogisticRegression(max_iter=300)), data_pca_256),
              ("LogisticRegression-C=1-L2_regularization-multinomial-PCA_256", LogisticRegression(fit_intercept=False, multi_class='multinomial', solver='newton-cg'), data_pca_256),
              ("RF-md=None-mf=30-mss=2-150_est-PCA_256", RandomForestClassifier(max_depth=None, max_features=30, min_samples_split=2, n_estimators=150, n_jobs=-1), data_pca_256),
              ("LinearSVC-C=1-squared_hinge_loss-L2_regularization-OvR-PCA_256", LinearSVC(), data_pca_256),
              #("LinearSVC-C=0p01-squared_hinge_loss-L2_regularization-OvR-PCA_256", LinearSVC(C=0.01), data_pca_256),
              ("SVC-C=1-poly_kernel-degree=3-PCA_256", SVC(kernel='poly', degree=3, probability=True), data_pca_256),  # prob True to support voting soft
              #("SVC-C=1-rbf_kernel-degree=2-PCA_256", SVC(kernel='rbf', degree=2), data_pca_256),
              #("SVC-C=1-rbf_kernel-degree=3-PCA_256", SVC(kernel='rbf', degree=3), data_pca_256),
              #("SVC-C=1-rbf_kernel-degree=4-PCA_256", SVC(kernel='rbf', degree=4), data_pca_256),
              #("SVC-C=1-rbf_kernel-degree=5-PCA_256", SVC(kernel='rbf', degree=5), data_pca_256)
    ]

    model_metrics = [fit_test_and_save_model(model_desc, model, data, unique_labels) for model_desc, model, data in models]

    # quick and dirty addition to try voting models - i should implement this better/reduce duplication if i use it again
    do_voting_classifier = True
    if do_voting_classifier:
        # using the indexing created by the final set of models above
        voting_models_all = [('ETC', models[0][1]), ('LR-multinomial', models[1][1]), ('RF', models[2][1]),
                             ('LinearSVC-CS', models[3][1]), ('LinearSVC-OvR', models[4][1])]
        voting_models_256 = [('ETC', models[5][1]), ('RF', models[7][1]), ('SVC-poly', models[9][1]),
                             ('LR-multinomial', models[6][1]), ('LinearSVC', models[8][1])]
        model_metrics.append(fit_test_and_save_model('Voting-256-hard', VotingClassifier(voting_models_256, voting='hard'), data_pca_256, unique_labels))
        model_metrics.append(fit_test_and_save_model('Voting-all-hard', VotingClassifier(voting_models_all, voting='hard'), data, unique_labels))

        # can't do voting=soft w/ LinearSVC - oh well, I could drop the model but I like it - i'll just do hard
        #model_metrics.append(fit_test_and_save_model('Voting-256-soft', VotingClassifier(voting_models_256, voting='soft'), data_pca_256, unique_labels))


    model_metrics_df = pd.DataFrame(model_metrics, columns=['Desc', 'Accuracy', 'Overall time', 'Train time', 'Notes'])
    print(section_print_delimiter)
    print("Metrics")
    print(model_metrics_df)
    metric_filename = 'model_metrics-{}.csv'.format(file_datetime_now())
    model_metrics_df.to_csv(metric_filename)
    print("Wrote {}.".format(metric_filename))


def generate_kaggle_predictions():
    """
    Generate Kaggle predictions for the specified models, by loading the pickled model and predicting. Note that
    I think I'll only run this a few times, with certain selected models, so I've left in a places where I need to
    just modify the data to use by modifying the source vs. making it more general/easier to use.
    """
    output_text_with_time("Loading test features at {}...")
    features_test = pd.read_csv('features_test.csv', header=None).values
    labels_test = pd.read_csv('labels_test.csv', header=None).values.ravel()
    print("Loaded {} features and {} labels at {}.".format(features_test.shape, labels_test.shape, datetime.datetime.now()))

    # For each run, update with pickle filenames, so we transform using the same rules as we used w/ the training set
    pca_64_filename = 'pca_transform_64_components-20170603-203014.pickle'
    pca_256_filename = 'pca_transform_256_components-20170603-203015.pickle'
    pca_64 = pickle.load(open(pca_64_filename, 'rb'))
    pca_256 = pickle.load(open(pca_256_filename, 'rb'))

    # scale and PCA transform the data
    X_scaled = StandardScaler().fit_transform(features_test)
    X_scaled_64 = pca_64.transform(X_scaled)
    X_scaled_256 = pca_256.transform(X_scaled)

    # Specify the models to use, and the corresponding data
    # like above, the last run I did was w/ just the voting classifiers, which is why the rest are commented out
    model_files = [
        ('Voting-all-hard-20170604-162450-model.pickle', X_scaled),
        ('Voting-256-hard-20170604-161810-model.pickle', X_scaled_256),

        # ('ETC-md=None-mf=75-mss=2-210_est-no_PCA-20170603-203022-model.pickle', X_scaled),
        # ('LogisticRegression-C=1-L2_regularization-OvR-no_PCA-20170603-203903-model.pickle', X_scaled),
        # ('LogisticRegression-C=1-L2_regularization-multinomial-no_PCA-20170603-203951-model.pickle', X_scaled),
        # ('RF-md=None-mf=60-mss=2-150_est-no_PCA-20170603-204010-model.pickle', X_scaled),
        # ('LinearSVC-C=1-squared_hinge_loss-L2_regularization-CS-no_PCA-20170603-204322-model.pickle', X_scaled),
        # ('LinearSVC-C=1-squared_hinge_loss-L2_regularization-OvR-no_PCA-20170603-204511-model.pickle', X_scaled),
        # ('LinearSVC-C=0p01-squared_hinge_loss-L2_regularization-OvR-no_PCA-20170603-204557-model.pickle', X_scaled),
        # ('ETC-md=None-mf=30-mss=2-210_est-PCA_256-20170603-204602-model.pickle', X_scaled_256),
        # ('MyLogisticRegression-C=1-max_iter=300-OvR-PCA_256-20170603-204708-model.pickle', X_scaled_256),
        # ('LogisticRegression-C=1-L2_regularization-multinomial-PCA_256-20170603-204732-model.pickle', X_scaled_256),
        # ('RF-md=None-mf=30-mss=2-150_est-PCA_256-20170603-204743-model.pickle', X_scaled_256),
        # ('LinearSVC-C=1-squared_hinge_loss-L2_regularization-OvR-PCA_256-20170603-204839-model.pickle', X_scaled_256),
        # ('LinearSVC-C=0p01-squared_hinge_loss-L2_regularization-OvR-PCA_256-20170603-204956-model.pickle', X_scaled_256),
        # ('SVC-C=1-poly_kernel-degree=3-PCA_256-20170603-205025-model.pickle', X_scaled_256),
        # ('SVC-C=1-rbf_kernel-degree=2-PCA_256-20170603-205107-model.pickle', X_scaled_256),
        # ('SVC-C=1-rbf_kernel-degree=3-PCA_256-20170603-205148-model.pickle', X_scaled_256),
        # ('SVC-C=1-rbf_kernel-degree=4-PCA_256-20170603-205230-model.pickle', X_scaled_256),
        # ('SVC-C=1-rbf_kernel-degree=5-PCA_256-20170603-205311-model.pickle', X_scaled_256)
    ]

    for model_file, X in model_files:
        model_desc = model_file.rstrip('-model.pickle')

        print(section_print_delimiter)
        print("Predicting using '{}' and data with shape {}.".format(model_desc, X.shape))
        output_text_with_time("Started at {}.")

        model = pickle.load(open(model_file, 'rb'))

        test_pred = model.predict(X)
        print("Generated predictions with shape {}.".format(test_pred.shape))

        test_results_for_output = pd.DataFrame({
            'Id': pd.Series(labels_test).astype(int),  # use labels as they go 1, 10, 100, etc.
            'Prediction': pd.Series(test_pred).apply(lambda pred: pred.split('.')[0].lstrip('0'))
        })

        print("Generated predictions with {} unique labels.".format(len(test_results_for_output['Prediction'].unique())))

        test_results_for_output.sort_values('Id', inplace=True)

        # save histogram for quick visual validation that we're not super-biased toward some labels
        test_results_for_output['Prediction'].astype(int).hist(bins=200)
        plt.savefig("{}-{}-test_hist".format(model_desc, file_datetime_now()))
        plt.close()

        filename = 'kaggle_test-{}.csv'.format(model_desc)
        test_results_for_output.to_csv('{}'.format(filename), index=False)

        print("Wrote {}.".format(filename))
        output_text_with_time("Finished at {}.")


def main():
    print(section_print_delimiter)

    # comment/uncomment to train models or predict the Kaggle set - ok to keep this as a comment (vs for ex
    # extending to respond to command line args) since I'll only be running the predict a few times and since
    # i'll already need to modify the source to specify which models I'm using to do the predictions

    #train_models()
    generate_kaggle_predictions()


if __name__ == '__main__':
    main()