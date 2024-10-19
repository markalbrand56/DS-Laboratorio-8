from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
import joblib
import pandas as pd

class PrediccionRequest():
    model = load_model('./models/modelo_final.keras')  # Modelo de red neuronal entrenado
    normalizador_x = joblib.load('./models/scaler_x.pkl')  # Normalizador de datos de entrada
    normalizador_y = joblib.load('./models/scaler_y.pkl')  # Normalizador de datos de salida

    label_encoder_animal = joblib.load('./models/le_animal.pkl')  # Label encoder para la variable animal
    label_encoder_city = joblib.load('./models/le_city.pkl')  # Label encoder para la variable city
    label_encoder_furniture = joblib.load('./models/le_furniture.pkl')  # Label encoder para la variable furniture

    def __init__(self, city: str, area: int, rooms: int, bathroom: int, parking_spaces: int, floor: int, animal: str, furniture: str, hoa: int, rent_amount: int, property_tax: int, fire_insurance: int):
        self.city = city
        self.area = area
        self.rooms = rooms
        self.bathroom = bathroom
        self.parking_spaces = parking_spaces
        self.floor = floor
        self.animal = animal
        self.furniture = furniture
        self.hoa = hoa
        self.rent_amount = rent_amount
        self.property_tax = property_tax
        self.fire_insurance = fire_insurance

    def to_dict(self):
        return {
            'city': self.city,
            'area': self.area,
            'rooms': self.rooms,
            'bathroom': self.bathroom,
            'parking spaces': self.parking_spaces,
            'floor': self.floor,
            'animal': self.animal,
            'furniture': self.furniture,
            'hoa (R$)': self.hoa,
            'rent amount (R$)': self.rent_amount,
            'property tax (R$)': self.property_tax,
            'fire insurance (R$)': self.fire_insurance,
        }
    
    def to_df(self):
        return pd.DataFrame([self.to_dict()])
    
    def predict(self):
        df = self.to_df()
        df['animal'] = self.label_encoder_animal.transform(df['animal'])
        df['city'] = self.label_encoder_city.transform(df['city'])
        df['furniture'] = self.label_encoder_furniture.transform(df['furniture'])
        df = self.normalizador_x.transform(df)
        
        prediction = self.model.predict(df)
        respuesta = self.normalizador_y.inverse_transform(prediction)
        
        return respuesta[0][0]

# Crear una instancia de la clase PrediccionRequest
prediccion = PrediccionRequest(
    city='Porto Alegre',
    area=100,
    rooms=3,
    bathroom=2,
    parking_spaces=1,
    floor=1,
    animal='acept',
    furniture='furnished',
    hoa=500,
    rent_amount=2000,
    property_tax=200,
    fire_insurance=100
)

# Realizar la predicción
print(prediccion.predict())