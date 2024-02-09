#!/usr/bin/env python
# coding: utf-8

# MLP Classifier

# In[1]:


from sklearn.neural_network import MLPClassifier
import time
from sklearn.metrics import f1_score, accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
import numpy as np
from tensorflow.keras.layers import *


# In[2]:


def build_mlp(X_train, y_train, X_test):
    start_time = time.time()
    
    mlp_classifier = MLPClassifier()
    mlp_classifier.fit(X_train, y_train)

    mlp_predictions = mlp_classifier.predict(X_test)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time of MLP: {:.2f} seconds".format(execution_time))
    return mlp_predictions


# In[3]:


def mlp_metrics(mlp_predictions, y_test):
    f_score = f1_score(y_test, mlp_predictions, average='micro')
    accuracy = accuracy_score(y_test, mlp_predictions)
    print(classification_report(y_test, mlp_predictions))
    print("MLP F-score:", f_score)
    print("MLP Accuracy:", accuracy)
    return f_score, accuracy


# In[4]:


def build_mlp_with_embedding(X_train, y_train, X_test, embedding_matrix, vocab_size,embed_dim=150,input_length=304):    
    model = Sequential()
    
    # Create and set the weights for the embedding layer
    embedding_layer = Embedding(vocab_size, embed_dim, input_length=input_length, 
                                 weights=[embedding_matrix], trainable=False)
    
    model.add(embedding_layer)
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=5, batch_size=64)
    
    mlp_predictions = model.predict(X_test)
    
    return mlp_predictions


# In[ ]:




