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
This model contains ensemble of SVM, PA, XGBoost, LSTM and BiLSTM as first classifier and then using stacking ensemble for inputs of final classifier

The functionality of each model are seprated in diffrente '.ipnyb' files. and then the scripts of them are in Scripts folder.

## Model 8:
- We drop 'half-true' datas and combine classes to 2 classes.
- Then we have full preproccesing on data.
- Then we tokenized and used Word2Vec on data.
- Stacking is by binary prediction of models.
- Embedding dim is 150
 