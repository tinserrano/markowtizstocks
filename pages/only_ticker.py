import streamlit as st
import yfinance as yf
from curl_cffi import requests
import pandas as pd
from datetime import datetime, timedelta

# Configurar el título de la aplicación
st.title("Descargar datos de un ticker")

# Campo para ingresar el ticker
ticker = st.text_input("Ingresa el símbolo del ticker:", "^SPX")

# Configurar fechas
fecha_fin = datetime.now()
fecha_inicio = fecha_fin - timedelta(days=365)  # Un año hacia atrás por defecto

# Opciones para seleccionar fechas
fecha_inicio_str = st.date_input("Fecha de inicio:", fecha_inicio)
fecha_fin_str = st.date_input("Fecha de fin:", fecha_fin)

# Botón para descargar datos
if st.button("Descargar datos"):
    # Crear sesión con curl_cffi
    session = requests.Session(impersonate="chrome")
    
    # Mostrar mensaje de carga
    with st.spinner(f'Descargando datos para {ticker}...'):
        # Descargar los datos
        df = yf.download(
            tickers=ticker,
            start=fecha_inicio_str.strftime('%Y-%m-%d'),
            end=fecha_fin_str.strftime('%Y-%m-%d'),
            auto_adjust=True,
            session=session
        )
        
        # Mostrar el DataFrame
        st.write(f"Datos para {ticker}:")
        st.dataframe(df)
        
        # Opción para descargar como CSV
        csv = df.to_csv().encode('utf-8')
        st.download_button(
            label="Descargar como CSV",
            data=csv,
            file_name=f"{ticker}_datos.csv",
            mime="text/csv",
        )




st.markdown("<br>", unsafe_allow_html=True)
'''    

    [![Linkedin](https://badgen.net/badge/Linkedin/Here?icon=https://simpleicons.now.sh/linkedin&label?color=black)](https://www.linkedin.com/in/martinepenas/)
'''