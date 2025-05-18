import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import datetime
from curl_cffi import requests as cffi_requests

from PIL import Image

# Manejo seguro de la imagen
try:
    image = Image.open("harry.png")
    st.image(image, caption="Harry Markowitz")
except FileNotFoundError:
    st.title("Cartera de Inversión - Frontera de Eficiencia - Markowitz")

'''
# Cartera de Inversion - Frontera de Eficiencia - Markowitz

La teoría de cartera eficiente de Markowitz supone que los inversores son racionales y buscan maximizar rendimientos minimizando riesgos, basándose en que estos riesgos pueden medirse mediante la variabilidad (volatilidad) de los rendimientos y que la diversificación adecuada permite optimizar la relación riesgo-rendimiento.

Para elegir el mejor portafolio entre una cantidad de portafolios posibles, cada uno con diferente rentabilidad y riesgo, se deben tomar dos decisiones por separado:

1. Determinación de un conjunto de carteras eficientes.

2. Selección de la mejor cartera del conjunto eficiente.

Si bien puedes elegir la cartera que desees, aqui te comentamos cuál sería la cartera eficiente en cuanto al Ratio Sharpe (rentabilidad / riesgo) 


&nbsp;&nbsp;&nbsp;&nbsp;
'''

'''
## Cartera de Inversion de Acciones
### Te recomendaremos una cartera de inversión entre acciones populares del mercado.
&nbsp;&nbsp;
'''

# Lista de acciones populares para elegir
stock_options = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", 
    "JNJ", "V", "PG", "DIS", "NFLX", "INTC", "CSCO", "KO", "PEP", "WMT", 
    "XOM", "CVX", "BAC", "HD", "VZ", "MRK", "PFE", "T", "ADBE", "CRM","CAAP", "GGAL", "BMA", "YPF"
]

a = st.multiselect("Elige tus acciones para el armado de la cartera (te mostrará datos de los últimos 2 años)",
                 stock_options)
q_sim = st.slider("Selecciona la cantidad de simulaciones", 4000, 12000, 6000)



# Definir fechas para el rango de 2 años
import datetime
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=2*365)  # Aproximadamente 2 años

# Formatear fechas como strings en formato YYYY-MM-DD
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')



def get_stock_data(tickers):
    """
    Descarga datos históricos para una lista de tickers usando yfinance.download.
    Enfoque probado que funciona en Colab.
    """
    try:
        # Mostrar qué tickers estamos procesando
        st.write(f"Obteniendo datos para: {', '.join(tickers)}")
        
        # Intentar crear una sesión con curl_cffi
        try:
            session = cffi_requests.Session(impersonate="chrome")
        except Exception as e:
            st.warning(f"No se pudo crear sesión con curl_cffi: {str(e)}")
            st.warning("Intentando descargar sin sesión personalizada...")
            session = None
    
        
        st.write(f"Período: {start_str} a {end_str}")
        
        # Intento principal con sesión personalizada
        if session is not None:
            try:
                data = yf.download(
                    tickers=' '.join(tickers),  # Unir tickers con espacios
                    start=start_str,
                    end=end_str,
                    group_by='ticker',
                    auto_adjust=True,
                    session=session
                )
                
            except Exception as e:
                st.warning(f"Error con sesión personalizada: {str(e)}")
                st.warning("Intentando sin sesión personalizada...")
                session = None
        
        # Intento alternativo sin sesión personalizada si el anterior falló
        if session is None:
            try:
                data = yf.download(
                    tickers=' '.join(tickers),
                    start=start_str,
                    end=end_str,
                    group_by='ticker',
                    auto_adjust=True
                )
                st.success("Descarga exitosa sin sesión personalizada")
            except Exception as e:
                st.error(f"Error en la descarga sin sesión: {str(e)}")
                return pd.DataFrame()  # Devolver DataFrame vacío si todo falla
        
        # Preparar DataFrame con precios de cierre
        close_data = pd.DataFrame()
        
        # Si solo hay un ticker, la estructura es diferente
        if len(tickers) == 1:
            ticker = tickers[0]
            if 'Close' in data.columns:
                close_data[ticker] = data['Close']
        else:
            # Para múltiples tickers, extraer los precios de cierre
            for ticker in tickers:
                if (ticker, 'Close') in data.columns:
                    close_data[ticker] = data[(ticker, 'Close')]
        
        # Verificar que tenemos datos
        if close_data.empty:
            st.warning("No se encontraron datos de cierre para los tickers proporcionados")
        else:
            st.success(f"Se obtuvieron datos para {len(close_data.columns)} de {len(tickers)} tickers")
        
        return close_data
        
    except Exception as e:
        st.error(f"Error general al descargar datos: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()
    

if st.button("Ya elegí mis acciones favoritas"):
    if not a:
        st.error("Por favor, selecciona al menos una acción.")
        st.stop()
    
    progress_text = st.empty()
    bar = st.progress(0)

    symbols = a
    
    try:
        # Mostrar progreso inicial
        progress_text.text("Configurando descarga de datos...")
        bar.progress(0.1)
        
        # Descargar datos usando yf.download directamente
        progress_text.text("Descargando datos de todas las acciones seleccionadas...")
        bar.progress(0.3)
        
        df_prices = get_stock_data(symbols)
        
        # Verificar si tenemos datos
        if df_prices.empty:
            st.error("No se pudieron obtener datos para las acciones seleccionadas.")
            st.stop()
            
        
        # Mostrar progreso
        progress_text.text("Procesando datos...")
        bar.progress(0.7)
        
        # Limpiar datos
        df_prices = df_prices.fillna(method='ffill').fillna(method='bfill')
        
        # Verificar suficientes datos
        if df_prices.shape[0] < 30:  # Necesitamos al menos 30 días para un buen análisis
            st.error("No hay suficientes datos históricos para realizar un análisis confiable.")
            st.stop()
            
        # Calcular rendimientos logarítmicos
        rendimientos = np.log(df_prices / df_prices.shift(1)).dropna()
        
        # Analizar cartera
        progress_text.text("Analizando carteras posibles...")
        bar.progress(0.8)
        
        rportafolio = []
        sdportafolio = []
        pesosportafolio = []
        numero_activos = len(rendimientos.columns)
        q = q_sim

        # Simulación Monte Carlo para encontrar cartera óptima
        with st.spinner("Realizando simulaciones para encontrar la cartera óptima..."):
            for x in range(q):
                # Generar pesos aleatorios
                pesos = np.random.random(numero_activos)
                pesos /= np.sum(pesos)
                pesosportafolio.append(pesos)
                
                # Calcular rendimiento esperado anualizado
                rportafolio.append(np.sum(rendimientos.mean() * pesos) * 252)
                
                # Calcular volatilidad anualizada
                matriz_cov = rendimientos.cov() * 252
                var = np.dot(pesos.T, np.dot(matriz_cov, pesos))
                sdportafolio.append(np.sqrt(var))

        # Crear DataFrame con resultados
        progress_text.text("Generando resultados...")
        bar.progress(0.9)
        
        diccionario = {"Rendimientos": rportafolio, "Volatilidad": sdportafolio}
        for contador, ticker in enumerate(rendimientos.columns.tolist()):
            diccionario["Peso Relativo -" + ticker] = [w[contador] for w in pesosportafolio]

        matrizportafolio = pd.DataFrame(diccionario)
        matrizportafolio["sharpe_ratio"] = matrizportafolio["Rendimientos"] / matrizportafolio["Volatilidad"]

        # Encontrar cartera con mejor ratio de Sharpe
        mejor_idx = matrizportafolio["sharpe_ratio"].argmax()
        maxvol = matrizportafolio["Volatilidad"].iloc[mejor_idx]
        maxrend = matrizportafolio["Rendimientos"].iloc[mejor_idx]
        
        # Completar barra de progreso
        progress_text.text("¡Análisis completado!")
        bar.progress(1.0)
        time.sleep(0.5)
        progress_text.empty()

        # Gráfico de la frontera eficiente
        fig4 = go.Figure(data=go.Scatter(
            x=matrizportafolio["Volatilidad"],
            y=matrizportafolio["Rendimientos"],
            mode='markers',
            marker=dict(
                size=5,
                color=matrizportafolio["sharpe_ratio"],
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig4.add_trace(go.Scatter(
            x=[maxvol],
            y=[maxrend],
            mode="markers", 
            marker=dict(color="red", symbol="star", size=12),
            name="Cartera Óptima"
        ))

        fig4.update_layout(
            title="Cartera Eficiente Markowitz",
            xaxis_title="Volatilidad (Riesgo)",
            yaxis_title="Rendimiento Esperado",
            width=2200,
            height=800
        )
        st.plotly_chart(fig4, theme="streamlit", use_container_width=True)

        # Obtener pesos de la cartera óptima
        optimo = matrizportafolio.loc[mejor_idx]
        optimochart = optimo.drop(["Rendimientos", "Volatilidad", "sharpe_ratio"], axis=0)
        optimochart = pd.DataFrame(optimochart)
        optimochart = optimochart.T
        
        # Gráfico de pie con la distribución de la cartera óptima
        fig3 = px.pie(
            values=optimochart.iloc[0,:], 
            names=optimochart.columns, 
            title="Distribución de Inversión según ratio Sharpe",
            hole=0.3
        )
        fig3.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig3, theme="streamlit")

        # Cálculos para comparación de rendimiento
        df_prices_normalized = df_prices / df_prices.iloc[0]
        
        # Calcular rendimientos de estrategias
        # 1. Cartera con pesos iguales
        df_partes_iguales = pd.DataFrame(
            df_prices_normalized.mean(axis=1), 
            columns=["partes_iguales"]
        )
        
        # 2. Cartera óptima
        pesos_optimos = []
        for col in df_prices.columns:
            peso_col = optimochart.iloc[0][f"Peso Relativo -{col}"] if f"Peso Relativo -{col}" in optimochart.columns else 0
            pesos_optimos.append(peso_col)
        
        # Asegurar que los pesos sumen 1
        pesos_optimos = np.array(pesos_optimos)
        if sum(pesos_optimos) > 0:
            pesos_optimos = pesos_optimos / sum(pesos_optimos)
        
        df_cartera_optima = pd.DataFrame(
            np.dot(df_prices_normalized, pesos_optimos),
            index=df_prices_normalized.index,
            columns=["cartera_optima"]
        )
        
        # Combinar para graficar
        combined_df = pd.concat([df_partes_iguales, df_cartera_optima], axis=1)
        
        # Añadir SPY para comparación
        try:
            # Crear una nueva sesión para SPY
            session = cffi_requests.Session(impersonate="chrome")
            
            # Obtener datos de SPY en el mismo período
            spy_data = yf.download(
                "SPY", 
                start=start_str, 
                end=end_date,
                session=session
            )
            
            if not spy_data.empty:
                spy_normalized = spy_data['Close'] / spy_data['Close'].iloc[0]
                spy_normalized = pd.DataFrame(spy_normalized, columns=["SPY"])
                
                # Asegurar que tienen el mismo índice
                combined_df = combined_df.merge(
                    spy_normalized, 
                    left_index=True, 
                    right_index=True, 
                    how='left'
                )
                combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
        except Exception as e:
            st.warning(f"No se pudieron obtener datos de comparación con SPY: {str(e)}")
        
        # Convertir a rendimiento porcentual
        combined_df = (combined_df - 1) * 100
        
        # Graficar comparación de rendimientos
        fig5 = px.line(
            combined_df, 
            x=combined_df.index, 
            y=combined_df.columns,
            labels={'value': 'Rendimiento (%)', 'variable': 'Estrategia'},
            title='Comparación de Rendimientos a lo largo del tiempo'
        )

        fig5.update_layout(
            xaxis_title='Fecha', 
            yaxis_title='Rendimiento (%)',
            width=2700, 
            height=600,
            legend_title_text='Estrategia'
        )
        
        # Renombrar leyenda para mayor claridad
        fig5.for_each_trace(lambda t: t.update(
            name={
                'partes_iguales': 'Cartera Pesos Iguales',
                'cartera_optima': 'Cartera Óptima Markowitz',
                'SPY': 'S&P 500 (SPY)'
            }.get(t.name, t.name)
        ))
        
        st.plotly_chart(fig5, theme="streamlit", use_container_width=True)

        # Calcular estadísticas de rendimiento y riesgo
        try:
            final_values = combined_df.iloc[-1]
            
            # Calcular volatilidades anualizadas
            volatilities = combined_df.pct_change().std() * np.sqrt(252)
            
            # Calcular ratios de Sharpe
            sharpe_ratios = final_values / volatilities
            
            # Crear tabla de resultados
            result_data = {
                'Estrategia': combined_df.columns,
                'Rendimiento (%)': final_values.values,
                'Volatilidad Anualizada (%)': volatilities.values * 100,
                'Ratio Sharpe': sharpe_ratios.values
            }
            
            results_df = pd.DataFrame(result_data)
            
            # Formatear la tabla
            results_df['Rendimiento (%)'] = results_df['Rendimiento (%)'].round(2)
            results_df['Volatilidad Anualizada (%)'] = results_df['Volatilidad Anualizada (%)'].round(2)
            results_df['Ratio Sharpe'] = results_df['Ratio Sharpe'].round(2)
            
            # Mostrar la tabla de resultados
            st.subheader("Comparación de Estrategias")
            
            # Renombrar para presentación
            results_df['Estrategia'] = results_df['Estrategia'].replace({
                'partes_iguales': 'Cartera Pesos Iguales',
                'cartera_optima': 'Cartera Óptima Markowitz',
                'SPY': 'S&P 500 (SPY)'
            })
            
            st.dataframe(results_df.set_index('Estrategia'), use_container_width=True)
            
            # Mostrar resultados individuales con colores
            for i, row in results_df.iterrows():
                strategy = row['Estrategia']
                rendimiento = row['Rendimiento (%)']
                volatilidad = row['Volatilidad Anualizada (%)']
                
                color = ":red:" if "SPY" in strategy else (":green:" if "Óptima" in strategy else ":blue:")
                st.write(f"Rendimiento {strategy}: {color}[{rendimiento:.2f}%]")
                st.write(f"Volatilidad {strategy}: {color}[{volatilidad:.2f}%]")
                
        except Exception as e:
            st.error(f"Error al calcular estadísticas comparativas: {str(e)}")

        st.title("")
        st.title("")
        '''
        ### Si quieres llevarte los % de cartera, te lo dejamos en formato tabla para que lo copies y pegues en tu excel, googlesheets, papelito, en la mano :smirk: "
        &nbsp;&nbsp;
        '''
        # Limpiar nombres de columnas para mejor presentación
        optimochart_display = optimochart.copy()
        optimochart_display.columns = [col.replace('Peso Relativo -', '') for col in optimochart_display.columns]
        
        # Mostrar la tabla de pesos de la cartera óptima
        st.dataframe(optimochart_display, use_container_width=True)
        
        # Ofrecer descarga de los resultados
        csv = optimochart_display.to_csv(index=False)
        st.download_button(
            label="Descargar cartera como CSV",
            data=csv,
            file_name="cartera_optima_acciones.csv",
            mime="text/csv"
        )

        '''
        &nbsp;

        ###### Disclaimer: La presente sólo tiene fines didácticos. No es para nada una recomendación de compra. Has tu propia investigación. DYOR
        &nbsp;&nbsp;
        '''

        st.title("")

        st.title("Contáctame")
        '''
            [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/tinserrano) 

            [![Medium](https://badgen.net/badge/Medium/Link?icon=https://simpleicons.now.sh/medium&label?color=black)](https://medium.com/@tinsonico) 
            
            [![Linkedin](https://badgen.net/badge/Linkedin/Here?icon=https://simpleicons.now.sh/linkedin&label?color=black)](https://www.linkedin.com/in/martinepenas/)
        '''
        st.markdown("<br>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ha ocurrido un error inesperado: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

else:
    st.write("Aqui aparecerá tu recomendación de cartera")