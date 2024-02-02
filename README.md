# fake-news-detection-using-ensemble-classifier
The perpouse of this repo is detecting fake news with an ensemble of deep learning models. This repo is related to a my researches on base of "Deep Ensemble Fake News Detection Model Using Sequential Deep Learning Technique". This research is in protected by my license!!

# Structure

**UsedPaper:** Base of "Deep Ensemble Fake News Detection Model Using Sequential Deep Learning Technique".

**liar_dataset:** The LIAR dataset is exist in "sample_data" folder. It downloades from public available link.

# Settings

- Programming Language: Python
- Libraries: Numpy, Pandas, tensorflow
- Dataset: LIAR

# Models:

## Model 1:
This model contains ensemble of NN models as first classifier.

## Model 4:
This model contains ensemble of SVM, LSTM and PA as first classifier and then using stacking ensemble for inputs of MLP classifier.

## Model 5:
This model contains ensemble of SVM, LSTM, PA and XGBoost as first classifier and then using stacking ensemble for inputs of MLP classifier.

The main diffrence of Model4 with Model 5 is in preproccesing. In Model 5 I used SMOTE for oversampling 'Real' news in datasets for equl distribution between classes.

Also I added more plots for classifiers in this one.

**Experiment History:** 





