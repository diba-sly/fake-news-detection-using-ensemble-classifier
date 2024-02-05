#!/usr/bin/env python
# coding: utf-8

# MLP Classifier

# In[1]:


from sklearn.neural_network import MLPClassifier
import time
from sklearn.metrics import f1_score, accuracy_score, classification_report


# In[ ]:


def build_mlp(X_train, y_train, X_test):
    start_time = time.time()
    
    mlp_classifier = MLPClassifier()
    mlp_classifier.fit(X_train, y_train)

    mlp_predictions = mlp_classifier.predict(X_test)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time of MLP: {:.2f} seconds".format(execution_time))
    return mlp_predictions


# In[2]:


def mlp_metrics(mlp_predictions, y_test):
    f_score = f1_score(y_test, mlp_predictions, average='micro')
    accuracy = accuracy_score(y_test, mlp_predictions)
    print(classification_report(y_test, mlp_predictions))
    print("MLP F-score:", f_score)
    print("MLP Accuracy:", accuracy)
    return f_score, accuracy


# In[ ]:




