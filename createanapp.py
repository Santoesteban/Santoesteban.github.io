# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 08:12:38 2023

@author: Hugo
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

import os
import tarfile
import urllib.request
#from combined_attributes_adder import CombineAttributeAdder



from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]





#https://github.com/Santoesteban/Santoesteban.github.io.git

def predict(data, model_name):
    model = joblib.load(f'{model_name}')
    pipeline= joblib.load('pipeline.sav')
    transformed_data = pipeline.transform(data)
    return model.predict(transformed_data)
# Cargar el modelo



DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
fetch_housing_data()
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


header = st.container()
dataset = st.container()
inputs = st.container()
modelTraining = st.container()


with header:
    st.title('Precio de las casas')
    

with dataset:
    st.header('Housing Dataset')
    st.text('Se muestran la información que existe en nuestra base de datos')
    housing = load_housing_data()
    st.write(housing)


with inputs:
    st.header('Inputs del modelo')
    st.text('Selecciona los inputs para poder predecir el precio de la casa de tus sueños')
    
    sel_col, disp_col= st.columns(2)
    
    longitude = st.number_input('longitud', min_value=-200, max_value= 0, value= -100, step=1)
    latitude = st.number_input('latitud', min_value=0, max_value= 60, value= 30, step=1)
    housing_median_age = st.number_input('hausing_median_age', min_value=0, max_value= 100, value= 20, step=1)
    location = sel_col.selectbox('¿En qué zona te gustaría?', options=["ISLAND","NEAR BAY", "NEAR OCEAN", "INLAND", "<1H OCEAN"], index = 0)
    total_rooms = st.number_input('Número de cuartos', min_value=1, max_value= 10, value= 1, step=1)
    total_bedrooms = st.number_input('Número de baños', min_value=1, max_value= 10, value= 1, step=1)
    population = st.number_input('población', min_value=1, max_value= 4000, value= 500, step=1)
    households = st.number_input('households', min_value=1, max_value= 2000, value= 500, step=1)
    median_income = st.number_input('medianincome', min_value=1, max_value= 2000, value= 500, step=1)
    
    data = [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households,median_income]
    
    
    data = pd.DataFrame(data)

    option = st.selectbox(
        '¿Qué método quieres usar?',
        ('Regresión Lineal', 'Random forest', 'decision tree'))





    if option == 'Regresión Lineal':
        st.write('You selected:', option)
        predict(data, 'l_regression.sav')
    if option == 'Random forest':
        st.write('You selected:', option)
        predict(data, 'ramf_regression.sav')
    if option == 'decision tree':
        st.write('You selected:', option)
        predict(data, 'dTree_regression.sav')






"""data es el conjunto de datos que se van a escribir en el programa

# Crear un input para ingresar un número
valor = st.number_input('Ingrese un número', min_value=0, max_value=100, value=50)

# Mostrar el valor ingresado
st.write('El número ingresado es:', valor)

data = pd.DataFrame(
    [
       {valor, "1": 7.5, "2": "430000"},

   ]
)
edited_df = st.experimental_data_editor(data)


favorite_command = edited_df.loc[edited_df["median_income"].idxmax()]["median_house_value"]
st.markdown(f"Your favorite command is **{favorite_command}** 🎈")

st.write(data)

option = st.selectbox(
    '¿Qué método quieres usar?',
    ('Regresión Lineal', 'Random forest', 'decision tree'))





if option == 'Regresión Lineal':
    st.write('You selected:', option)
    predict(data, 'l_regression.sav')
    print("random")
if option == 'Random forest':
    st.write('You selected:', option)
    print("random")
if option == 'decision tree':
    st.write('You selected:', option)
    print("random")

"""
