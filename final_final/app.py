import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import json
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. ԷՋԻ ԿԱՐԳԱՎՈՐՈՒՄՆԵՐ (PAGE SETUP)
# ==========================================
st.set_page_config(
    page_title="Solar Forecast Pro", 
    page_icon="☀️", 
    layout="wide"
)

st.title("☀️ Արևային Էներգիայի Կանխատեսման Համակարգ")
st.markdown("Այս հավելվածը օգտագործում է **Կասկադային Մեքենայական Ուսուցում (XGBoost)**՝ ապագա հզորությունը կանխատեսելու համար:")

# ==========================================
# 2. ՖՈՒՆԿՑԻԱՆԵՐ ԵՎ ՄՈԴԵԼՆԵՐԻ ԲԵՌՆՈՒՄ
# ==========================================
# @st.cache_resource-ը ապահովում է, որ մոդելները բեռնվեն միայն 1 անգամ և կայքը արագ աշխատի
@st.cache_resource
def load_models():
    temp_model = joblib.load('temperature_model.pkl')
    power_model = joblib.load('power_model.pkl')
    with open('config.json', 'r') as f:
        config = json.load(f)
    return temp_model, power_model, config

temp_model, power_model, config = load_models()

def fetch_weather(lat, lon, days):
    """Քաշում է եղանակը Open-Meteo API-ից ըստ նշված օրերի"""
    hours = days * 24
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, 
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "shortwave_radiation"],
        "wind_speed_unit": "ms", "timezone": "auto", "forecast_hours": hours
    }
    response = requests.get(url, params=params).json()
    hourly = response['hourly']
    
    df = pd.DataFrame({
        'DATE_TIME': pd.to_datetime(hourly['time']),
        'IRRADIATION': hourly['shortwave_radiation'],
        'T2M': hourly['temperature_2m'],
        'WS10M': hourly['wind_speed_10m'],
        'RH2M': hourly['relative_humidity_2m'],
        'PRECTOTCORR': hourly['precipitation']
    })
    return df

# ==========================================
# 3. ԿՈՂԱՅԻՆ ՎԱՀԱՆԱԿ (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Կառավարման Վահանակ")
st.sidebar.markdown("Մուտքագրեք կայանի տվյալները.")

# Վերցնում ենք դեֆոլտ կոորդինատները մեր config.json-ից
default_lat = float(config['location']['latitude'])
default_lon = float(config['location']['longitude'])

lat = st.sidebar.number_input("Լայնություն (Latitude)", value=default_lat, format="%.4f")
lon = st.sidebar.number_input("Երկայնություն (Longitude)", value=default_lon, format="%.4f")

# Սլայդեր՝ օրեր ընտրելու համար
forecast_days = st.sidebar.slider("Կանխատեսման հորիզոն (Օրեր)", min_value=1, max_value=7, value=3)

# Գլխավոր կոճակը
run_forecast = st.sidebar.button("🚀 Սկսել Կանխատեսումը", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("Համակարգը ավտոմատ կերպով կմիանա արբանյակային եղանակի սերվերներին և կկիրառի Արհեստական Ինտելեկտի մոդելները:")


# ==========================================
# 4. ԳԼԽԱՎՈՐ ԷԿՐԱՆ ԵՎ ՀԱՇՎԱՐԿ
# ==========================================
if run_forecast:
    with st.spinner('Միանում ենք օդերևութաբանական կայանին և կատարում հաշվարկներ...'):
        
        # 1. Քաշում ենք եղանակը
        df_future = fetch_weather(lat, lon, forecast_days)
        
        # 2. Կասկադային Գուշակություն
        X_weather = df_future[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
        
        # Փուլ 1. Վիրտուալ Սենսոր (Ջերմաստիճան)
        pred_temp = temp_model.predict(X_weather)
        
        # Փուլ 2. Գլխավոր Գեներատոր (Հոսանք)
        X_full = X_weather.copy()
        X_full.insert(1, 'MODULE_TEMPERATURE', pred_temp)
        df_future['PREDICTED_DC_POWER'] = np.maximum(0, power_model.predict(X_full))
        
        # 3. Արագ Ցուցանիշներ (Metrics)
        st.subheader("📊 Հիմնական Ցուցանիշներ")
        col1, col2, col3 = st.columns(3)
        
        max_power = df_future['PREDICTED_DC_POWER'].max()
        total_energy = df_future['PREDICTED_DC_POWER'].sum() / 1000 # վերածում ենք կՎտ-ի մոտավոր
        
        col1.metric("Պիկային Հզորություն", f"{max_power:,.0f} Վտ")
        col2.metric("Ընդհանուր Սպասվող Էներգիա", f"{total_energy:,.1f} կՎտժ")
        col3.metric("Կանխատեսվող Ժամեր", f"{len(df_future)} ժամ")

        st.markdown("---")
        
        # 4. Ինտերակտիվ Գրաֆիկ (Plotly)
        st.subheader("📈 Գեներացիայի Ինտերակտիվ Գրաֆիկ")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_future['DATE_TIME'], 
            y=df_future['PREDICTED_DC_POWER'],
            mode='lines+markers',
            name='Հզորություն (Վտ)',
            line=dict(color='#00b894', width=3),
            fill='tozeroy', # Ներկում է գծի տակը
            fillcolor='rgba(0, 184, 148, 0.2)'
        ))
        
        fig.update_layout(
            title=f"Առաջիկա {forecast_days} օրվա արտադրության կանխատեսում",
            xaxis_title="Ամսաթիվ և Ժամ",
            yaxis_title="DC Power (Վտ)",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 5. Աղյուսակ և Ներբեռնում
        st.subheader("📋 Կանխատեսման Աղյուսակ")
        st.dataframe(df_future[['DATE_TIME', 'IRRADIATION', 'T2M', 'PREDICTED_DC_POWER']], use_container_width=True)
        
        # CSV Ներբեռնման կոճակ
        csv = df_future.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Ներբեռնել արդյունքները (CSV)",
            data=csv,
            file_name=f'Forecast_{forecast_days}_days.csv',
            mime='text/csv',
        )

else:
    # Սկզբնական դատարկ էկրանի հաղորդագրությունը
    st.info("👈 Խնդրում ենք ձախ վահանակից սեղմել **«Սկսել Կանխատեսումը»** կոճակը՝ արդյունքները տեսնելու համար։")