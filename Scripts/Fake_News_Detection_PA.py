#!/usr/bin/env python
# coding: utf-8

# PA Classifier

# In[15]:


from sklearn.linear_model import PassiveAggressiveClassifier
import time
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV


# In[9]:


def train_pa_gridshearch():
    start_time = time.time()
    param_grid = {
    'C': [0.1,0.5,],              # Regularization parameter
    'fit_intercept': [True, False],
    'max_iter': [500,1000],          # Maximum number of iterations
    'tol': [1e-6, 1e-7]              # Tolerance for stopping criterion
    }
   # Create a PassiveAggressiveClassifier
    pa = PassiveAggressiveClassifier()

    grid_search = GridSearchCV(pa, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_valid_resampled, y_train_valid_resampled)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)
    


# In[10]:


def train_pa(X_train,y_train,X_test):
    start_time = time.time()
    
    pa_model = PassiveAggressiveClassifier(C = 0.01, random_state = 42, max_iter=1000, early_stopping=True, tol=1e-9)
    pa_model.fit(X_train, y_train)
    
    pa_train_valid_predictions = pa_model.predict(X_train)
    pa_predictions = pa_model.predict(X_test)
    y_scores = pa_model.decision_function(X_test)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time of PA: {:.2f} seconds".format(execution_time))
    return pa_predictions,pa_train_valid_predictions,y_scores


# In[11]:


def pa_metrics(pa_predictions, y_test):
    accuracy = sum(pa_predictions == y_test) / len(y_test)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test,
                                                                         pa_predictions, average='binary')
    print("PA - accuracy:", accuracy)
    print("PA - Precision:", precision)
    print("PA - Recall:", recall)
    print("PA - F-score:", fscore)
    return accuracy, precision, recall, fscore


# In[12]:


def pa_confusion_matrix_plot(pa_predictions, y_test):
    cm = confusion_matrix(y_test, pa_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - PA')
    plt.show()


# In[14]:


def pa_roc_plot(pa_predictions, y_test):
    # Compute the true positive rate (tpr) and false positive rate (fpr) using roc_curve
    pa_fpr, pa_tpr, pa_thresholds = roc_curve(y_test, pa_predictions)
    pa_roc_auc = auc(pa_fpr, pa_tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(pa_fpr, pa_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % pa_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) -PA')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:




