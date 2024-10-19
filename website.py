from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
import joblib
import pandas as pd

modelo_final = load_model('./models/modelo_final.keras')
normalizador_x = joblib.load('./models/scaler_x.pkl')
normalizador_y = joblib.load('./models/scaler_y.pkl')

label_encoder_animal = joblib.load('./models/le_animal.pkl')
label_encoder_city = joblib.load('./models/le_city.pkl')
label_encoder_furniture = joblib.load('./models/le_furniture.pkl')

def dev_prediction(modelo, muestra: pd.DataFrame):

    # muestra es un DataFrame con una sola fila que tiene los datos de entrada
    row = np.array(muestra)

    print("Row: ", row)

    prediction = modelo.predict(row)

    respuesta_escala_original = normalizador_y.inverse_transform(prediction)

    print("Predicción: ", respuesta_escala_original)

    return (prediction[0][0], respuesta_escala_original[0][0])

# city,area,rooms,bathroom,parking spaces,floor,animal,furniture,hoa (R$),rent amount (R$),property tax (R$),fire insurance (R$),total (R$)


test = {
    'city': 'Porto Alegre',
    'area': 100,
    'rooms': 3,
    'bathroom': 2,
    'parking spaces': 1,
    'floor': 1,
    'animal': 'acept',
    'furniture': 'furnished',
    'hoa (R$)': 500,
    'rent amount (R$)': 2000,
    'property tax (R$)': 200,
    'fire insurance (R$)': 100,
}

# Convertir a DataFrame
test_df = pd.DataFrame([test])

test_df['animal'] = label_encoder_animal.transform(test_df['animal'])
test_df['city'] = label_encoder_city.transform(test_df['city'])
test_df['furniture'] = label_encoder_furniture.transform(test_df['furniture'])

# Normalizar los datos
test_df = normalizador_x.transform(test_df)

# Realizar la predicción
print(dev_prediction(modelo_final, test_df))