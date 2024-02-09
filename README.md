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

The functionality of each model are seprated in diffrente '.ipnyb' files. and then the scripts of them are in Scripts folder, so in Model6 I imported the Scripts models to use the functionality of them.

## Model 6:
- We drop 'half-true' datas and combine classes to 2 classes.
- Then we have full preproccesing on data.
- Then we tokenized and used Word2Vec on data.
- Stacking is by binary prediction of models.
- Embedding dim is 1500
The functionality of each model are seprated in diffrente '.ipnyb' files. and then the scripts of them are in Scripts folder, so in Model6 I imported the Scripts models to use the functionality of them.


## Model 7:
- We combine classes to 2 classes. then we merge 50% of 'half-true' data in class 1 and rest of it in class 2.
- We used full preproccess.
- We tokenized and used Word2Vec.
- We use y_scores for stacking but it wasn't helpfull so we used binary predictions.

## Model 8:
- We combined classes to 2 classes. The 2 of them as True and rest of them as False.
- We used full preproccess.
- We tokenized and used Word2Vec.
- We used binary prediction for stacking.

## Model 9:
- We drop 'half-true' datas and combine classes to 2 classes.
- Then we have full preproccesing on data.
- Then we tokenized and used Word2Vec on data.
- Stacking is by binary prediction of models.
- Embedding dim is 150
  
## Model 10:
- We drop 'half-true' datas and combine classes to 2 classes.
- We did TFIDF on data for machine learning models.
- We did Word2Vec on data for deep learning models.
- Then we have full preproccesing on data.
- Stacking is by binary prediction of models.
- Embedding dim is 150




