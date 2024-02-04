#!/usr/bin/env python
# coding: utf-8

# Read data and prepare them

# In[101]:


import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# In[102]:


#Global constants
tokenizer = Tokenizer()
vectorizer = TfidfVectorizer()
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 150


# In[103]:


def combine_classes(dataset):
    dataset['label']=[1 if x=="true"or x=="mostly-true" else 0 for x in dataset[1]]
    
    #Dealing with empty datapoints for metadata columns - subject, speaker, job, state,affiliation, context
    meta = []
    for i in range(len(dataset)):
      subject = dataset[3][i]
      if subject == 0:
          subject = 'None'

      speaker =  dataset[4][i]
      if speaker == 0:
          speaker = 'None'

      job =  dataset[5][i]
      if job == 0:
          job = 'None'

      state =  dataset[6][i]
      if state == 0:
          state = 'None'

      affiliation =  dataset[7][i]
      if affiliation == 0:
          affiliation = 'None'

      context =  dataset[13][i]
      if context == 0 :
          context = 'None'

      meta.append(str(subject) + ' ' + str(speaker) + ' ' + str(job) + ' ' + str(state) + ' ' + str(affiliation) + ' ' + str(context)) #combining all the meta data columns into a single column
  
    #Adding cleaned and combined metadata column to the dataset
    dataset[14] = meta
    dataset["sentence"] = dataset[14].astype('str')+" "+dataset[2]
    
    #Dropping unwanted columns
    dataset = dataset.drop(labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14] ,axis=1)
    dataset.dropna()
    return dataset


# In[104]:


def data_preprocessing(dataset):
    preprocessed_texts = []
    for text in dataset:
        # convert to lowercase
        text = text.lower()    
        # tokenize text
        tokens = word_tokenize(text)
        # remove punctuation and irrelevant characters
        filtered_tokens = [token for token in tokens if token.isalnum()]
        # remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in filtered_tokens if not token in stop_words]
        # lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        filtered_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        # stem tokens
        stemmer = PorterStemmer()
        filtered_tokens = [stemmer.stem(token) for token in filtered_tokens]
        # join tokens back into string
        preprocessed_text = ' '.join(filtered_tokens)
        preprocessed_texts.append(preprocessed_text)
    return preprocessed_texts


# In[105]:


def data_preprocessing_without_stem(dataset):
    preprocessed_texts = []
    for text in dataset:
        text = text.lower()
        text = str(text).replace(r'\.\.+','.') #replace multiple periods with a single one
        text = str(text).replace(r'\.','.') #replace periods with a single one
        text = str(text).replace(r'\s\s+',' ') #replace multiple white space with a single one
        text = str(text).replace("\n", "") #removing line breaks
        
        # tokenize text
        tokens = word_tokenize(text)
        # remove punctuation and irrelevant characters
        filtered_tokens = [token for token in tokens if token.isalnum()]
        # remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in filtered_tokens if not token in stop_words]
        
        # lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        filtered_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        # join tokens back into string
        preprocessed_text = ' '.join(filtered_tokens)
        preprocessed_texts.append(preprocessed_text)
    return preprocessed_texts


# In[106]:


def do_preprocessing(data_train, data_valid, data_test,preproccess_type="all"):
    if preproccess_type=="all":
        data_train['sentence'] = data_preprocessing(data_train['sentence'])
        data_valid['sentence'] = data_preprocessing(data_valid['sentence'])
        data_test['sentence'] = data_preprocessing(data_test['sentence'])
        print("data_preprocessing done!")
    else:
        data_train['sentence'] = data_preprocessing_without_stem(data_train['sentence'])
        data_valid['sentence'] = data_preprocessing_without_stem(data_valid['sentence'])
        data_test['sentence'] = data_preprocessing_without_stem(data_test['sentence'])
        print("data_preprocessing_without_stem done!")
    data_train.head(5)
    return data_train, data_valid, data_test


# In[115]:


def data_tokenizer(X_train, X_test, data_set_all):
#     tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data_set_all['sentence'])
    
    vocab_size =len(tokenizer.word_index) + 1
    
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    return X_train,X_test, vocab_size


# In[108]:


def tfidf_vectorizer(X_train, X_test):
#     vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    vocab_size = len(vectorizer.vocabulary_) + 1  # Vocabulary size for word embedding
    
    # Prepare the input sequences for the LSTM model
    X_train_sequences = pad_sequences(X_train_vectorized.toarray(),
                                            maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post')
    X_test_sequences = pad_sequences(X_test_vectorized.toarray(),
                                            maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post')
    print("Vectorizing done!")
    return X_train_vectorized, X_test_vectorized, vocab_size, X_train_sequences, X_test_sequences


# In[118]:


def word2vec_vectorizer(X_train, X_test, data_set_all):
    
    word2vec_model = Word2Vec(sentences=[text.split() for text in data_set_all['sentence']],
                              vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)
    vectorized_data = []

    for texts in [X_train, X_test]:
        data_vectors = []
        for text in texts:
            text_vectors = [word2vec_model.wv[word] for word in text if word in word2vec_model.wv]
            if len(text_vectors) > 0:
                data_vectors.append(np.mean(text_vectors, axis=0))
            else:
                data_vectors.append(np.zeros(word2vec_model.vector_size))
        vectorized_data.append(data_vectors)

    X_train_vectors, X_test_vectors = vectorized_data
    
    # Create embedding matrix
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]
            
    # Prepare the input sequences for the LSTM model
    X_train_sequences = pad_sequences(X_train_vectors,
                                            maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post')
    X_test_sequences = pad_sequences(X_test_vectors,
                                            maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post')
    
    print("Word2Vec vectorizing done!")
    return X_train_vectors, X_test_vectors, embedding_matrix,X_train_sequences,X_test_sequences


# In[120]:


def tokenize_vectorize(data_set, data_train, data_test):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data_set['sentence'])
    
    # Convert text to sequences
    X_train = tokenizer.texts_to_sequences(data_train['sentence'])
    X_test = tokenizer.texts_to_sequences(data_test['sentence'])
    
    
    # Pad sequences
    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

    # Prepare target labels
    y_train = data_train['label'].values
    y_test = data_test['label'].values
    
    # Create Word2Vec embeddings
    word2vec = Word2Vec(sentences=[text.split() for text in data_set['sentence']],
                        vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)

    # Create embedding matrix
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if word in word2vec.wv:
            embedding_matrix[i] = word2vec.wv[word]
    vocab_size =len(tokenizer.word_index) + 1
    return embedding_matrix,X_train, y_train, X_test, y_test,vocab_size


# In[110]:


# def word2vec_embed(X_train,X_valid,X_test):
    
#     combined_data = X_train + X_valid + X_test

#     # Word2Vec embedding
#     word2vec = Word2Vec(sentences=combined_data, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)

#     # Split the embedded data back into training, validation, and test sets
#     X_train_word2vec = [word2vec.wv[phrase] for phrase in X_train]
#     X_valid_word2vec = [word2vec.wv[phrase] for phrase in X_valid]
#     X_test_word2vec = [word2vec.wv[phrase] for phrase in X_test]

#     # Create embedding matrix
#     embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
#     for word, i in tokenizer.word_index.items():
#         if word in word2vec.wv:
#             embedding_matrix[i] = word2vec.wv[word]
#     print("Word2Vec embedding done!")
#     return embedding_matrix,X_train_word2vec,X_valid_word2vec,X_test_word2vec


# In[111]:


def oversampling_data(X,y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)

    print("Before SMOTE:", Counter(y))
    print("After SMOTE:", Counter(y_resampled))
    return X_resampled, y_resampled


# In[112]:


def tokenize_vectorize_data(data_set):
    # Tokenization
    tokenizer.fit_on_texts(your_text_data)
    sequences = tokenizer.texts_to_sequences(your_text_data)

    # Word2Vec
    word2vec = Word2Vec(sentences=sequences, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)


# In[113]:


def read_data():
    data_train = pd.read_csv("sample_data/train.tsv", sep="\t", header=None)
    data_valid = pd.read_csv("sample_data/valid.tsv", sep="\t", header=None)
    data_test = pd.read_csv("sample_data/test.tsv", sep="\t", header=None)
    data_train.head(3)
    return data_train, data_valid, data_test


# In[114]:


def plot_data_length(data_set):
    #Analyzing length of sentences in training data to decide on MAX_LENGTH variable, which is required for mlp and deep_leaner
    sent_len = []
    for sent in data_set['sentence']:
      sent_len.append(len(sent))

    fig = plt.figure(figsize =(10, 7))
    plt.boxplot(sent_len)
    plt.show()

    sent_len = [i for i in sent_len if i<=500] #Excluding the outliers
    fig2 = plt.figure(figsize =(10, 7))
    plt.hist(sent_len, 5)
    plt.show()

