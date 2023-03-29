# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:21:50 2023

@author: Hugo
"""

import joblib

def predict(data, model_name):
    model = joblib.load(f'{model_name}')
    pipeline= joblib.load('iris_pipeline.sav')
    transformed_data = pipeline.transform(data)
    return model.predict(transformed_data)

