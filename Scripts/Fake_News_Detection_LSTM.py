#!/usr/bin/env python
# coding: utf-8

# LSTM classifier

# In[27]:


import re
#from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import time


# In[28]:


# Constants
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 150


# In[29]:


def pad_sequence_data(X_train,X_valid,X_test):
    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_valid = pad_sequences(X_valid, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    return X_train,X_valid,X_test


# In[36]:


def train_lstm(vocab_size, embedding_matrix, X_train, y_train, epoches=10,batch_size=128):
    start_time = time.time()
    
    model = Sequential()
    model.add(Input(shape=(MAX_SEQUENCE_LENGTH,)))
    model.add(Embedding(vocab_size, EMBEDDING_DIM, weights = [embedding_matrix],
                        input_length = MAX_SEQUENCE_LENGTH, trainable=False, name = 'embeddings'))
    model.add(LSTM(128, return_sequences=True,name='lstm_layer1'))
    model.add(LSTM(64, return_sequences=True,name='lstm_layer2'))
    model.add(GlobalMaxPool1D())
    #model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

    model.summary()
    
    history = model.fit(X_train, y_train, epochs = epoches, batch_size=batch_size, validation_split=0.2)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time of LSTM: {:.2f} seconds".format(execution_time))

    return model, history


# In[37]:


def predict_lstm(lstm_model, X_train, X_test):
    lstm_train_predictions = (lstm_model.predict(X_train) > 0.4).astype(int).flatten()
    lstm_predictions = (lstm_model.predict(X_test) > 0.4).astype(int).flatten()
    return lstm_train_predictions, lstm_predictions


# In[32]:


def lstm_metrics(lstm_predictions, y_test):
    accuracy = sum(lstm_predictions == y_test) / len(y_test)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, lstm_predictions, average='binary')

    print("LSTM - Accuracy:", accuracy)
    print("LSTM - Precision:", precision)
    print("LSTM - Recall:", recall)
    print("LSTM - F-score:", fscore)
    return accuracy, precision, recall, fscore


# In[33]:


def lstm_loss_plot(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss in LSTM')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show();


# In[34]:


def lstm_confusion_matrix_plot(lstm_predictions, y_test):
    cm = confusion_matrix(y_test, lstm_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - LSTM')
    plt.show()


# In[35]:


def lstm_roc_plot(lstm_predictions, y_test):
    # Compute the true positive rate (tpr) and false positive rate (fpr) using roc_curve
    lstm_fpr, lstm_tpr, lstm_thresholds = roc_curve(y_test, lstm_predictions)
    lstm_roc_auc = auc(lstm_fpr, lstm_tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(lstm_fpr, lstm_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % lstm_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) -LSTM')
    plt.legend(loc="lower right")
    plt.show()

