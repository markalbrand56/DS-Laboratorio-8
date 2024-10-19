from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid


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

        # Transformar las variables categóricas
        df['animal'] = self.label_encoder_animal.transform(df['animal'])
        df['city'] = self.label_encoder_city.transform(df['city'])
        df['furniture'] = self.label_encoder_furniture.transform(df['furniture'])

        # Normalizar los datos
        df = self.normalizador_x.transform(df)
        
        # Realizar la predicción y desnormalizar el resultado
        prediction = self.model.predict(df)
        respuesta = self.normalizador_y.inverse_transform(prediction)
        
        return respuesta[0][0]
    
    # App layout
    @staticmethod
    def app():
        df = pd.read_csv('./data/houses_to_rent_v2.csv')
        df['floor'] = df['floor'].str.extract('(\d+)').fillna(0).astype(int)

        st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

            .custom-font {
                font-family: 'Montserrat', sans-serif;
            }
                    
            h1.custom-font{
                font-size: 32px;
                color: #d0ece7;
            }
            .st-emotion-cache-15hul6a {
                background-color: #138d75 !important;
                color: white !important;
                border: none;
                padding: 10px 20px !important;
                border-radius: 5px;
                font-size: 16px !important;
                font-weight: bold;
                cursor: pointer !important;
            }

            .st-emotion-cache-15hul6a {
                background-color: #117a65;
            }
                    
            .result-box {
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-family: 'Roboto', sans-serif;
                color: #333;
                font-size: 24px;
                font-weight: bold;
            }
                    
            </style>
            """, unsafe_allow_html=True)


        with st.sidebar:
            selected = option_menu(None, ["Predecir", "Tendencias", "Acerca del dataset"],
                icons=['clipboard2-data-fill', 'bar-chart', 'database'],
                menu_icon="cast", default_index=0, orientation="vertical",
                styles={
                    "container": {"padding": "0!important", "background-color": "#000"},
                    "icon": {"font-size": "25px"}, 
                    "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#0e6655"},
                    "nav-link-selected": {"background-color": "#16a085"},
                }
            )

        if selected == 'Predecir':
            st.markdown('<h1 class="custom-font">Predicción de Precio de Alquiler</h1>', unsafe_allow_html=True)
            if not df.empty and all(col in df.columns for col in ['area', 'rooms', 'bathroom', 'parking spaces', 'floor', 'hoa (R$)', 'rent amount (R$)', 'property tax (R$)', 'fire insurance (R$)']):
                st.write("Complete los siguientes campos para predecir el precio de alquiler de una propiedad.")
                with st.form("property_form"):
                    area = st.number_input("Área en m²", min_value=float(df['area'].min()), max_value=float(df['area'].max()))
                    habitaciones = st.number_input("Número de habitaciones", min_value=float(df['rooms'].min()), max_value=float(df['rooms'].max()))
                    banos = st.slider("Número de baños", min_value=float(df['bathroom'].min()), max_value=float(df['bathroom'].max()), step=1.0)
                    parking_spaces = st.slider("Número de espacios de estacionamiento", min_value=float(df['parking spaces'].min()), max_value=float(df['parking spaces'].max()), step=1.0)
                    piso = st.slider("Número de piso", min_value=int(df['floor'].min()), max_value=int(df['floor'].max()), step=1)
                    animal = st.selectbox("¿Se permiten mascotas?", ("acept", "not acept"))
                    muebles = st.selectbox("Mobiliario", ("furnished", "not furnished"))
                    hoa = st.number_input("Cuota de HOA (R$)", min_value=float(df['hoa (R$)'].min()), max_value=float(df['hoa (R$)'].max()))
                    renta = st.number_input("Monto de renta (R$)", min_value=float(df['rent amount (R$)'].min()), max_value=float(df['rent amount (R$)'].max()))
                    impuesto_propiedad = st.number_input("Impuesto a la propiedad (R$)", min_value=float(df['property tax (R$)'].min()), max_value=float(df['property tax (R$)'].max()))
                    seguro_incendio = st.number_input("Seguro contra incendios (R$)", min_value=float(df['fire insurance (R$)'].min()), max_value=float(df['fire insurance (R$)'].max()))
                    ciudad = st.selectbox("Ciudad", df['city'].unique(), key='ciudad')                
                    submit_button = st.form_submit_button("Predecir Precio de Alquiler")

                if submit_button:
                    if area and habitaciones and banos and ciudad and animal and muebles:
                        data = pd.DataFrame([{
                            'city': ciudad,
                            'area': area,
                            'rooms': habitaciones,
                            'bathroom': banos,
                            'parking spaces': parking_spaces,
                            'floor': piso,
                            'animal': animal,
                            'furniture': muebles,
                            'hoa (R$)': hoa,
                            'rent amount (R$)': renta,
                            'property tax (R$)': impuesto_propiedad,
                            'fire insurance (R$)': seguro_incendio
                        }])

                        data['animal'] = PrediccionRequest.label_encoder_animal.transform(data['animal'])
                        data['city'] = PrediccionRequest.label_encoder_city.transform(data['city'])
                        data['furniture'] = PrediccionRequest.label_encoder_furniture.transform(data['furniture'])

                        data_normalized = PrediccionRequest.normalizador_x.transform(data)
                        
                        # Realizar la predicción y desnormalizar el resultado
                        prediccion = PrediccionRequest.model.predict(data_normalized)
                        prediccion_desnormalizada = PrediccionRequest.normalizador_y.inverse_transform(prediccion)
                        prediccion = prediccion_desnormalizada[0][0]
                        st.markdown(f'<div class="result-box">El precio estimado de alquiler es: R${prediccion:,.2f} </div>', unsafe_allow_html=True)
                    else:
                        st.error("Todos los campos son obligatorios.")

        elif selected == 'Tendencias':
            st.markdown('<h1 class="custom-font">Tendencias de Precio de Alquiler</h1>', unsafe_allow_html=True)
            df = pd.read_csv('./data/houses_to_rent_v2.csv')
            feature_list = ['city', 'area', 'rooms', 'bathroom', 'parking spaces', 'floor', 'animal', 'furniture', 'hoa (R$)', 'rent amount (R$)', 'property tax (R$)', 'fire insurance (R$)']
            for feature in feature_list:
                st.write(f"### Relación entre {feature} y el Precio Total de Alquiler")
                if df[feature].dtype == 'object':
                    fig, ax = plt.subplots()
                    avg_price = df.groupby(feature)['total (R$)'].mean().reset_index()
                    sns.barplot(x=feature, y='total (R$)', data=avg_price, ax=ax)
                    ax.set_title(f"Promedio del Precio Total según {feature}")
                    st.pyplot(fig)
                else:
                    fig, ax = plt.subplots()
                    sns.regplot(x=feature, y='total (R$)', data=df, ax=ax, scatter_kws={'alpha':0.5})
                    ax.set_title(f"Relación entre {feature} y Precio Total (Regresión Lineal)")
                    st.pyplot(fig)

        elif selected == 'Acerca del dataset':
            st.markdown('<h1 class="custom-font">Acerca del Dataset</h1>', unsafe_allow_html=True)
            st.write("El dataset consiste en alquileres de viviendas en Brasil en el 2020.")
            st.write("Este dataset incluye las siguientes características:")
            st.write("- Ciudad")
            st.write("- Área")
            st.write("- Número de habitaciones")
            st.write("- Número de baños")
            st.write("- Número de espacios de estacionamiento")
            st.write("- Piso")
            st.write("- Se aceptan mascotas")
            st.write("- Mobiliario")
            st.write("- Cuota de HOA")
            st.write("- Monto de renta")
            st.write("- Impuesto a la propiedad")
            st.write("- Seguro contra incendios")
            st.write("- Precio total")
            df = pd.read_csv('./data/houses_to_rent_v2.csv')
            AgGrid(df)



if __name__ == "__main__":
    PrediccionRequest.app()
