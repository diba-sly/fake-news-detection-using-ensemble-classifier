#!/usr/bin/env python
# coding: utf-8

# SVM classifier

# In[20]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# In[21]:


def train_svm(x_train, y_train, x_test):
    start_time = time.time()

    svm_model = SVC(C=10, gamma='scale', kernel='poly')
    svm_model.fit(x_train, y_train)
    svm_train_valid_predictions = svm_model.predict(x_train)
    svm_predictions = svm_model.predict(x_test)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time of SVM: {:.2f} seconds".format(execution_time))
    return svm_predictions, svm_train_valid_predictions


# In[22]:


def svm_gridseach(X_train, y_train, X_test):
    start_time = time.time()

    param_grid = {
        'C': [1.0, 10.0],  # Regularization parameter
        'kernel': ['poly', 'rbf'],  # Kernel type
        'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf'
    }

    svm_model = SVC()

    # Define the Grid Search with cross-validation
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='f1_micro')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best parameters of SVM:", best_params)
    
    svm_train_valid_predictions = grid_search.predict(X_train)
    svm_predictions = grid_search.predict(X_test)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time of SVM: {:.2f} seconds".format(execution_time))
    return svm_predictions,svm_train_valid_predictions


# In[23]:


def svm_metrics(svm_predictions, y_test):
    accuracy = sum(svm_predictions == y_test) / len(y_test)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test,
                                                                         svm_predictions, average='binary')
    print("SVM - accuracy:", accuracy)
    print("SVM - Precision:", precision)
    print("SVM - Recall:", recall)
    print("SVM - F-score:", fscore)
    return accuracy, precision, recall, fscore


# In[24]:


def svm_confusion_matrix_plot(svm_predictions, y_test):
    cm = confusion_matrix(y_test, svm_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - SVM')
    plt.show()


# In[25]:


def svm_roc_plot(svm_predictions, y_test):
    # Compute the true positive rate (tpr) and false positive rate (fpr) using roc_curve
    svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, svm_predictions)
    svm_roc_auc = auc(svm_fpr, svm_tpr)

    plt.figure()
    plt.plot(svm_fpr, svm_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % svm_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) -SVM')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:




