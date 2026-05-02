import pandas as pd
import requests
import joblib
import json
import streamlit as st
import os

# Ստեղծել տվյալների պանակը, եթե այն գոյություն չունի
os.makedirs('./data', exist_ok=True)

@st.cache_resource
def load_models_and_config():
    try:
        temp_model = joblib.load('./data/temperature_model.pkl')
        power_model = joblib.load('./data/power_model.pkl')
        with open('./data/config.json', 'r') as f:
            config = json.load(f)
        return temp_model, power_model, config
    except Exception as e:
        return None, None, None

def fetch_future_weather(lat, lon, days):
    hours = days * 24
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, 
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "shortwave_radiation"],
        "wind_speed_unit": "ms", "timezone": "auto", "forecast_hours": hours
    }
    hourly = requests.get(url, params=params).json()['hourly']
    df = pd.DataFrame({
        'DATE_TIME': pd.to_datetime(hourly['time']),
        'IRRADIATION': [x / 1000.0 for x in hourly['shortwave_radiation']], 
        'T2M': hourly['temperature_2m'],
        'WS10M': hourly['wind_speed_10m'],
        'RH2M': hourly['relative_humidity_2m'],
        'PRECTOTCORR': hourly['precipitation']
    })
    return df

@st.cache_data
def fetch_nasa_annual(lat, lon, year):
    # Քաշում ենք GHI, DNI և DHI pvlib-ի ֆիզիկայի համար
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    url = (f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
           f"parameters=ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,WS10M"
           f"&community=RE&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON")
    resp = requests.get(url).json()
    df = pd.DataFrame(resp['properties']['parameter']).reset_index().rename(columns={'index': 'DATE_TIME'})
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%Y%m%d%H')
    df.replace(-999.0, 0.0, inplace=True) # Բացակայող արևը սարքում ենք 0
    
    # Փոխում ենք անունները pvlib-ի ստանդարտին համապատասխան
    df = df.rename(columns={
        'ALLSKY_SFC_SW_DWN': 'ghi',
        'ALLSKY_SFC_SW_DNI': 'dni',
        'ALLSKY_SFC_SW_DIFF': 'dhi',
        'T2M': 'temp_air',
        'WS10M': 'wind_speed'
    })
    # pvlib-ը պահանջում է TimeZone, NASA-ն տալիս է UTC
    df = df.set_index('DATE_TIME').tz_localize('UTC')
    return df