import numpy as np
import pandas as pd
from google.colab import files
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
uploaded = files.upload()
train =pd.read_csv('train.csv',error_bad_lines=False, engine="python")
#this is revised data after the sliding window

uploaded = files.upload()
test =pd.read_csv('train.csv',error_bad_lines=False, engine="python")

# get X_train and y_train from csv files
X_train = train.drop(['subject', 'Activity'], axis=1)
y_train = train.Activity
# get X_test and y_test from test csv file
X_test = test.drop(['subject', 'Activity'], axis=1)
y_test = test.Activity
#print('X_train and y_train : ({},{})'.format(X_train.shape, y_train.shape))
#print('X_test  and y_test  : ({},{})'.format(X_test.shape, y_test.shape))
labels=['LAYING', 'SITTING','STANDING','WALKING','WALKING_DOWNSTAIRS','WALKING_UPSTAIRS']
import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=matplotlib.colors.Colormap):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "blue")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize=True, \
                 print_cm=True, cm_cmap=plt.cm.Greens):
    
    # to store results at various phases
    results = dict()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results['predicted'] = y_pred
    # calculate overall accuracty of the model
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    # store accuracy in results
    results['accuracy'] = accuracy
    print('---------------------')
    print('|      Accuracy      |')
    print('---------------------')
    print('\n    {}\n\n'.format(accuracy))
    # confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm: 
        print('--------------------')
        print('| Confusion Matrix |')
        print('--------------------')
        print('\n {}'.format(cm))
    # plot confusin matrix
    plt.figure(figsize=(8,8))
    plt.grid(b=False)
    plot_confusion_matrix(cm, classes=class_labels, normalize=True, title='Normalized confusion matrix by Honglin Bao', cmap = cm_cmap)
    plt.show()
    
    # get classification report
    print('-------------------------')
    print('| Classifiction Report |')
    print('-------------------------')
    classification_report = metrics.classification_report(y_test, y_pred)
    # store report in results
    results['classification_report'] = classification_report
    print(classification_report)
    
    # add the trained  model to the results
    results['model'] = model
    
    return results
    
def print_grid_search_attributes(model):
    # Estimator that gave highest score among all the estimators formed in GridSearch
    print('--------------------------')
    print('|      Best Estimator     |')
    print('--------------------------')
    print('\n\t{}\n'.format(model.best_estimator_))


    # parameters that gave best results while performing grid search
    print('--------------------------')
    print('|     Best parameters     |')
    print('--------------------------')
    print('\tParameters of best estimator : \n\n\t{}\n'.format(model.best_params_))


    #  number of cross validation splits
    print('---------------------------------')
    print('|   No of CrossValidation sets   |')
    print('--------------------------------')
    print('\n\tTotal numbre of cross validation sets: {}\n'.format(model.n_splits_))


    # Average cross validated score of the best estimator, from the Grid Search 
    print('--------------------------')
    print('|        Best Score       |')
    print('--------------------------')
    print('\n\tAverage Cross Validate scores of best estimator : \n\n\t{}\n'.format(model.best_score_))

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV  
    
# start Grid search
#parameters = {'C':[0.01, 0.1, 1, 10, 20, 30], 'penalty':['l2','l1']}
#log_reg = linear_model.LogisticRegression()
#log_reg_grid = GridSearchCV(log_reg, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
#log_reg_grid_results =  perform_model(log_reg_grid, X_train, y_train, X_test, y_test, class_labels=labels)

#parameters = {'C':[0.125, 0.5, 1, 2, 8, 16]}
#lr_svc = LinearSVC(tol=0.00005)
#lr_svc_grid = GridSearchCV(lr_svc, param_grid=parameters, n_jobs=-1, verbose=1)
#lr_svc_grid_results = perform_model(lr_svc_grid, X_train, y_train, X_test, y_test, class_labels=labels)

#from sklearn.svm import SVC
#parameters = {'C':[2,8,16],\
#              'gamma': [ 0.0078125, 0.125, 2]}
#rbf_svm = SVC(kernel='rbf')
#rbf_svm_grid = GridSearchCV(rbf_svm,param_grid=parameters, n_jobs=-1)
#rbf_svm_grid_results = perform_model(rbf_svm_grid, X_train, y_train, X_test, y_test, class_labels=labels)

#from sklearn.tree import DecisionTreeClassifier
#parameters = {'max_depth':np.arange(4,10,1)}
#dt = DecisionTreeClassifier()
#dt_grid = GridSearchCV(dt,param_grid=parameters, n_jobs=-1)
#dt_grid_results = perform_model(dt_grid, X_train, y_train, X_test, y_test, class_labels=labels)
#print_grid_search_attributes(dt_grid_results['model'])

from sklearn.ensemble import RandomForestClassifier
params = {'n_estimators': np.arange(5,101,20), 'max_depth':np.arange(3,12,2)}
rfc = RandomForestClassifier()
rfc_grid = GridSearchCV(rfc, param_grid=params, n_jobs=-1)
rfc_grid_results = perform_model(rfc_grid, X_train, y_train, X_test, y_test, class_labels=labels)
print_grid_search_attributes(rfc_grid_results['model'])
