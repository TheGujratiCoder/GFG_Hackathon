from django.http import HttpResponse
from django.shortcuts import render
from django.core.cache import cache

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import xgboost as xgb

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def about(request):
    return render(request, 'about.html')

def index(request):
    return render(request, 'index.html')

# Perform data preprocessing and generate necessary statistics

df = pd.read_csv("./templates/medals.csv")
df1=df
del df['country_code']
del df['country_3_letter_code']
All_Athlete_URL=set()

for x in df['athlete_url']:
  All_Athlete_URL.add(x)

del df['athlete_url']

df['athlete_full_name']=df.groupby(['discipline_title','medal_type'])['athlete_full_name'].transform(lambda x: ', '.join(str(i) for i in x))
df=df.drop_duplicates(subset=['discipline_title','slug_game','medal_type'])
df['athlete_full_name'].fillna('',inplace=True)

# 4 Knowing the Names of all participants participated till now
def participant(request):
    df = pd.read_csv("./templates/medals.csv")
    All_Participants=Get_Count(df['athlete_full_name'])
    l1=list(All_Participants.keys())
    l2=list(All_Participants.values())
    # print("Number Of Times a Participant has Participated : ")
    # return l1
    return render(request, 'participant.html', {'l1':l1})


### 3 Finding Number of Participants in All Sports
def Get_Count(x):
  d=dict()
  for i in x:
    d[i]=d.get(i,0)+1
  return d
def get_sports_types(request):
    # df = pd.read_csv("./templates/medals.csv")
    All_Sports=Get_Count(df['discipline_title'])
    Sports_Types=list(All_Sports.keys())
    Sports_Types_Count=list(All_Sports.values())
    Sports_Types = {
        'Sports_Types': Sports_Types
    }
    return render(request,'game.html',Sports_Types)


# 5 Finding Number of Countries Participated
def country(request):
    All_Countries=Get_Count(df['country_name'])
    # print("All Countries Participated Till Now and its frequency is : ")
    c=list(All_Countries.keys())
    f=list(All_Countries.values())
    # content={'c':c , 'f': f}

    return render(request, 'country.html', {'c': c ,'f':f})

def train_svm(request):
    # Model training and evaluation using SVM
    # ...
    # Load the medals dataset
    df = pd.read_csv("./templates/medals.csv")  
    # Select relevant features and target variable
    features=['discipline_title', 'event_gender', 'participant_type']
    target='medal_type'

    # Convert categorical variables to numerical using LabelEncoder
    encoder=LabelEncoder()
    df_encoded=df[features].apply(encoder.fit_transform)

    # Split the dataset into training and testing sets
    X_train,X_test,y_train,y_test=train_test_split(df_encoded,df[target],test_size=0.2,random_state=42)

    # Scale the features
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    # Initialize and train the SVM classifier
    clf=SVC(kernel='rbf',random_state=42)  # Customize the kernel as needed
    clf.fit(X_train_scaled,y_train)

    # Predict on the test set
    y_pred=clf.predict(X_test_scaled)
    return render(request, 'svm.html', { 't':y_test , 'p' :y_pred })


def train_random_forest(request):
    # Model training and evaluation using Random Forest

    df = pd.read_csv("./templates/medals.csv")  

    # Select relevant features and target variable
    features=['discipline_title','event_gender','participant_type']
    target='medal_type'

    # Convert categorical variables to numerical using one-hot encoding
    df_encoded=pd.get_dummies(df[features],drop_first=True)

    # Split the dataset into training and testing sets
    X_train,X_test,y_train,y_test=train_test_split(df_encoded,df[target],test_size=0.2,random_state=42)

    # Train the Random Forest Classifier
    clf=RandomForestClassifier(random_state=42)
    clf.fit(X_train,y_train)

    # Predict on the test set
    y_pred=clf.predict(X_test)
    return render(request, 'svm.html', { 't':y_test , 'p' :y_pred })


def train_xgboost(request):
    # Model training and evaluation using XGBoost
    # ...

    # Load the medals dataset
    # df=pd.read_csv('medals.csv')

    # Select relevant features and target variable
    features=['discipline_title','event_gender','participant_type']
    target='medal_type'

    # Encode the target variable to numerical values
    encoder=LabelEncoder()
    df[target]=encoder.fit_transform(df[target])

    # Convert categorical variables to numerical using LabelEncoder
    df_encoded=df[features].apply(encoder.fit_transform)

    # Split the dataset into training and testing sets
    X_train,X_test,y_train,y_test=train_test_split(df_encoded,df[target],test_size=0.2,random_state=42)

    # Initialize and train the XGBoost classifier
    clf=xgb.XGBClassifier(random_state=42)  # Customize parameters as needed
    clf.fit(X_train,y_train)

    # Predict on the test set
    y_pred=clf.predict(X_test)

    # Evaluate the model
    # print(classification_report(y_test,y_pred))
    # return classification_report(y_test, y_pred)
    return render(request, 'xgboost.html', { 't':y_test , 'p' :y_pred })


def train_mlp(request):
    # Model training and evaluation using MLP
    # ...
    # Load the medals dataset
    # df=pd.read_csv('medals.csv')

    # Select relevant features and target variable
    features=['discipline_title', 'event_gender', 'participant_type']
    target='medal_type'

    # Convert categorical variables to numerical using LabelEncoder and OneHotEncoder
    encoder=LabelEncoder()
    df_encoded=df[features].apply(encoder.fit_transform)
    onehot_encoder=OneHotEncoder(sparse=False)
    df_encoded=pd.DataFrame(onehot_encoder.fit_transform(df_encoded),columns=onehot_encoder.get_feature_names_out(features))

    # Split the dataset into training and testing sets
    X_train,X_test,y_train,y_test=train_test_split(df_encoded, df[target],test_size=0.2,random_state=42)

    # Initialize and train the MLP classifier
    clf=MLPClassifier(hidden_layer_sizes=(100, 100),random_state=42)  # Customize the hidden_layer_sizes as needed
    clf.fit(X_train,y_train)

    # Predict on the test set
    y_pred=clf.predict(X_test)

    # Evaluate the model
    # print(classification_report(y_test,y_pred))
    # return classification_report(y_test, y_pred)
    return render(request, 'svm.html', { 't':y_test , 'p' :y_pred })


def train_lstm(request):
    # Model training and evaluation using LSTM
    # ...
    # Load the medals dataset
    # df=pd.read_csv('medals.csv')

    # Select relevant features and target variable
    features=['discipline_title','event_gender','participant_type']
    target='medal_type'

    # Encode the target variable to numerical values
    encoder=LabelEncoder()
    df[target]=encoder.fit_transform(df[target])

    # Preprocess text data
    text_data=df['discipline_title']+' '+df['event_gender']+' '+ df['participant_type']

    # Tokenize the text data
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(text_data)
    sequences=tokenizer.texts_to_sequences(text_data)
    vocab_size=len(tokenizer.word_index) + 1

    # Pad sequences to have the same length
    max_length=max([len(seq) for seq in sequences])
    padded_sequences=pad_sequences(sequences,maxlen=max_length)

    # Split the dataset into training and testing sets
    X_train,X_test,y_train,y_test=train_test_split(padded_sequences,df[target],test_size=0.2,random_state=42)

    # Build the LSTM model
    model=Sequential()
    model.add(Embedding(vocab_size,100,input_length=max_length))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # Train the model
    model.fit(X_train,y_train,epochs=10,batch_size=32,verbose=1)

    # Evaluate the model
    y_pred_probs=model.predict(X_test)
    y_pred=[1 if prob >= 0.5 else 0 for prob in y_pred_probs]
    # print(classification_report(y_test,y_pred))
    # return classification_report(y_test, y_pred)
    return render(request, 'svm.html', { 't':y_test , 'p' :y_pred })






import pickle
from django.shortcuts import render
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the LSTM model (Assuming the model file is named 'lstm_model.h5')
model = load_model('lstm_model.h5')

# Function to preprocess the user input and make predictions
def medal(request):
    if request.method == 'POST':
        discipline_title = request.POST.get('discipline_title', '')
        event_gender = request.POST.get('event_gender', '')
        participant_type = request.POST.get('participant_type', '')

        # Load the tokenizer (Assuming the tokenizer file is named 'tokenizer.pkl')
        with open('tokenizer.pkl', 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)

        # Preprocess the user input similar to the model training data
        text_data = discipline_title + ' ' + event_gender + ' ' + participant_type

        # Tokenize and pad the user input
        sequences = tokenizer.texts_to_sequences([text_data])
        max_length = 100  # Specify the maximum sequence length here (must be the same as during model training)
        padded_sequences = pad_sequences(sequences, maxlen=max_length)

        # Make predictions using the loaded LSTM model
        y_pred_probs = model.predict(padded_sequences)
        predicted_class = 1 if y_pred_probs[0][0] >= 0.5 else 0

        # Map the predicted class to medal type
        medal_type = 'GOLD' if predicted_class == 1 else 'SILVER'

        return render(request, 'medal.html', {'medal_type': medal_type})
    else:
        return render(request, 'medal.html')
