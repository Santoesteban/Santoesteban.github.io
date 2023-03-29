# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 08:47:56 2023

@author: Hugo
"""

#import altair as alt
#import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import streamlit as st
from sklearn import datasets

from predictioniris import predict



iris = datasets.load_iris()

df = pd.DataFrame(iris['data'], columns = iris['feature_names'])
df['target']= pd.Series(iris['target'], name = 'target values')
df['target_name'] = df['target'].replace([0,1,2],['iris-' + species for species in iris['target_names'].tolist()])
df = df.head()



def home_page() -> None:
    st. title('Iris dataset')
    with st.expander('Show raw data'):
        st.write(df)
    #st.header('General information')
    #selected_species = st.radio('Select species', ['Setosa', 'Versicolor', 'Virginica'])
    #show_description(selected_species)
    
    st.header('Iris Type: ')
    col1, col2 = st.columns(2)
    with col1:
        st.header('Petal: ')
        Petal_length = st.slider('Petal length', value=1.0, min_value=0.0, max_value = 6.9, step=0.1)
        petal_width =  st.slider('Petal width', value=1.0, min_value=0.0, max_value = 2.5, step=0.1)
        st.header('Sepal: ')
        Sepal_length = st.slider('Sepal length', value=2.0, min_value=0.0, max_value = 7.9, step=0.1)
        sepal_width =  st.slider('Sepal width', value=2.0, min_value=0.0, max_value = 4.4, step=0.1)
        
        
        data = pd.DataFrame({'Petal length': [Petal_length],
                             'Petal width': [petal_width],
                             'Sepal length': [Sepal_length],
                             'Sepal width': [sepal_width],                        
                             })
    with col2:
        model = st.radio("Model", ["Logistic Regression", "Support Vector Machine", "Decision tree", "Voting classifier"])
        
    if st.button ('Toca para predecir la flor'):
        
        if model == 'Logistic Regression':
            result = predict(data, 'log_regression.sav')
        elif model == 'Support Vector Machine':
            result = predict(data, 'SVC.sav')
        elif model == 'Decision tree':
            result = predict(data, 'tree.sav')
        elif model == 'Voting classifier':
            result = predict(data, 'voting.sav')
            
        if result == 0:
            result = "Setosa"
        if result == 1:
            result = "Versicolor"
        if result == 2:
            result = "Virginica"
    
        st.text(f'lA FLOR ES: {result}')
  
home_page()