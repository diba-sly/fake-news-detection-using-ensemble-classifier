# Fake News Detection Using Ensemble Machine Learning and Deep Learning Models
The perpouse of this repo is detecting fake news with an ensemble of deep learning models. 
This repo is related to a my researches "Improving F-Score for Fake News Detection Using Machine Learning Techniques" on base of "Deep Ensemble Fake News Detection Model Using Sequential Deep Learning Technique".

# Structure

**UsedPaper:** Based of "Deep Ensemble Fake News Detection Model Using Sequential Deep Learning Technique".

**liar_dataset:** The LIAR dataset is exist in "sample_data" folder. It downloades from public available link.

# Settings

- Programming Language: Python
- Libraries: Numpy, Pandas, tensorflow
- Dataset: LIAR

# Models:

## Model 4:
This model contains ensemble of SVM, LSTM and PA as first classifier and then using stacking ensemble for inputs of MLP classifier.

## Model 5:
This model contains ensemble of SVM, LSTM, PA and XGBoost as first classifier and then using stacking ensemble for inputs of MLP classifier.

The main diffrence of Model4 with Model 5 is in preproccesing. In Model 5 I used SMOTE for oversampling 'Real' news in datasets for equl distribution between classes.

Also I added more plots for classifiers in this one.

## Model 6:

This model contains ensemble of SVM, PA, XGBoost, LSTM and BiLSTM as first classifier and then using stacking ensemble for inputs of MLP classifier.
the main diffrence of this Model6 is that the machine learnings models (SVM, PA and XGBoost) are vectorized just by TFIDF and then oversampling with SMOTE. but the deep learnings model (LSTM and BiLSTM) are tokenized and vectorized by Word2Vec. This is beacuse of the effects of preprocesses on result of each model.

The functionality of each model are seprated in diffrente '.ipnyb' files. and then the scripts of them are in Scripts folder, so in Model6 I imported the Scripts models to use the functionality of them.
**Experiment History:** 





