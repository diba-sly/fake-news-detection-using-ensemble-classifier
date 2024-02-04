#!/usr/bin/env python
# coding: utf-8

# BiLSTM classifier

# In[19]:


import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import time


# In[20]:


# Constants
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 150


# In[27]:


def train_bilstm(vocab_size,X_train,y_train,epochs=10,batch_size=128):

    start_time = time.time()
    loss_values = []

    bilstm_model = Sequential()
    bilstm_model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    bilstm_model.add(Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.25, dropout=0.25)))
    bilstm_model.add(Bidirectional(LSTM(units=64, recurrent_dropout=0.1, dropout=0.1)))
    bilstm_model.add(Dense(units=32, activation='relu'))
    bilstm_model.add(Dense(1, activation='sigmoid'))
    bilstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    bilstm_model.summary()
    
    history = bilstm_model.fit(X_train, y_train, epochs = epochs, batch_size=batch_size, validation_split=0.2)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time of BiLSTM: {:.2f} seconds".format(execution_time))
    return bilstm_model, history


# In[22]:


def predict_bilstm(bilstm_model, X_train, X_test):
    bilstm_train_predictions = (bilstm_model.predict(X_train) > 0.3).astype(int).flatten()
    bilstm_predictions = (bilstm_model.predict(X_test) > 0.3).astype(int).flatten()
    return bilstm_train_predictions, bilstm_predictions


# In[23]:


def bilstm_metrics(bilstm_predictions, y_test):
    accuracy = sum(bilstm_predictions == y_test) / len(y_test)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, bilstm_predictions, average='binary')

    print("BiLSTM - Accuracy:", accuracy)
    print("BiLSTM - Precision:", precision)
    print("BiLSTM - Recall:", recall)
    print("BiLSTM - F-score:", fscore)
    return accuracy,recision, recall, fscore


# In[24]:


def bilstm_loss_plot(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss in BiLSTM')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show();


# In[25]:


def bilstm_confusion_matrix_plot(bilstm_predictions, y_test):
    cm = confusion_matrix(y_test, bilstm_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - BiLSTM')
    plt.show()


# In[26]:


def bilstm_roc_plot(bilstm_predictions, y_test):
    # Compute the true positive rate (tpr) and false positive rate (fpr) using roc_curve
    bilstm_fpr, bilstm_tpr, bilstm_thresholds = roc_curve(y_test, bilstm_predictions)
    bilstm_roc_auc = auc(bilstm_fpr, bilstm_tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(bilstm_fpr, bilstm_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % bilstm_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) -BiLSTM')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:




