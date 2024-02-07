#!/usr/bin/env python
# coding: utf-8

# XGBoost classifier

# In[14]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import time
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve


# In[12]:


def train_xgboost_search(X_train, y_train, X_test, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
                        
    # Define the parameter grid to search over
    param_grid={
    'objective': ['binary:logistic'],
    'eval_metric': ['logloss'],
    'eta': [0.01, 0.1],  # learning rate
    'max_depth': [15, 20, 25],  # maximum depth of a tree
    'subsample': [0.2, 0.5],  # subsample ratio of the training instances
    'colsample_bytree': [0.5, 0.8]  # subsample ratio of columns when constructing each tree
    }

    num_rounds = 300  # number of boosting rounds (iterations)

    xgb_model = xgb.XGBClassifier()

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='f1', cv=10)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best Parameters found by Grid Search:", best_params)

    model = xgb.train(best_params, dtrain, num_rounds)
    y_pred = model.predict(dtest)
    y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

    return y_pred_binary


# In[19]:


def train_xgboost(X_train, y_train, X_test, y_test, params, rounds=300, threshold=0.4):
    start_time = time.time()
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
#     params = {
#     'objective': 'binary:logistic',  # or 'multi:softmax' for multi-class classification
#     'eval_metric': 'logloss',  # or other appropriate evaluation metric
#     'eta': 0.01,  # learning rate
#     'max_depth': 20,  # maximum depth of a tree
#     'subsample': 0.2,  # subsample ratio of the training instances
#     'colsample_bytree': 0.8  # subsample ratio of columns when constructing each tree
#     }
    
    num_rounds = rounds  # number of boosting rounds (iterations)     
    model = xgb.train(params, dtrain, num_rounds)
    
    
    y_pred = model.predict(dtest)
    y_train_valid_pred = model.predict(dtrain)
    
    xgb_pred_binary = [1 if p >= threshold else 0 for p in y_pred]
    xgb_train_valid_pred_binary = [1 if p >= threshold else 0 for p in y_train_valid_pred]
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time of XGBoost: {:.2f} seconds".format(execution_time))
    
    return xgb_pred_binary, xgb_train_valid_pred_binary, y_pred, y_train_valid_pred


# In[1]:


def xgboost_cross_validation(X_train, y_train, params, rounds=300, fold=5):
    dtrain = xgb.DMatrix(X_train, label=y_train)
   
#     params = {
#     'objective': 'binary:logistic',  # or 'multi:softmax' for multi-class classification
#     'eval_metric': 'logloss',  # or other appropriate evaluation metric
#     'eta': 0.01,  # learning rate
#     'max_depth': 20,  # maximum depth of a tree
#     'subsample': 0.2,  # subsample ratio of the training instances
#     'colsample_bytree': 0.8  # subsample ratio of columns when constructing each tree
#     }
    
    num_rounds = rounds  # number of boosting rounds (iterations)     
    model = xgb.train(params, dtrain, num_rounds)
    
    kfold = KFold(n_splits=fold, shuffle=True, random_state=42)
    
    def f1_scoring(estimator, X, y):
        y_pred = estimator.predict(xgb.DMatrix(X))
        return f1_score(y, y_pred)
    
    results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=f1_scoring)
    
    print(f'F1 Score: {results.mean():.4f} (+/- {results.std():.4f})')


# In[8]:


def xgb_metrics(xgb_predictions, y_test):
    accuracy = sum(xgb_predictions == y_test) / len(y_test)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test,
                                                                         xgb_predictions, average='binary')
    print("XGBoost - accuracy:", accuracy)
    print("XGBoost - Precision:", precision)
    print("XGBoost - Recall:", recall)
    print("XGBoost - F-score:", fscore)
    return accuracy, precision, recall, fscore


# In[9]:


def xgb_confusion_matrix_plot(xgb_predictions, y_test):
    cm = confusion_matrix(y_test, xgb_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - XGBoost')
    plt.show()


# In[10]:


def xgb_roc_plot(xgb_predictions, y_test):
    # Compute the true positive rate (tpr) and false positive rate (fpr) using roc_curve
    xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(y_test, xgb_predictions)
    xgb_roc_auc = auc(xgb_fpr, xgb_tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(xgb_fpr, xgb_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % xgb_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - XGBoost')
    plt.legend(loc="lower right")
    plt.show()


# In[18]:


def xgb_f1_threshold_plot(y_test, y_pred):
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_test, y_pred > threshold) for threshold in thresholds]
    # Plot F1 score vs. threshold
    plt.figure()
    plt.plot(thresholds, f1_scores)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold - XGBoost')
    plt.show()


# In[ ]:


def xgb_learning_plot():
    # Plot learning curve
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, scoring='f1', cv=5)

    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.xlabel('Number of training examples')
    plt.ylabel('F1 Score')
    plt.title('Learning Curve - XGBoost')
    plt.legend()
    plt.show()

