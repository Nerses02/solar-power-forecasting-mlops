import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import json
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime

# ==========================================
# 1. ԷՋԻ ԿԱՐԳԱՎՈՐՈՒՄՆԵՐ ԵՎ ՀԻՇՈՂՈՒԹՅՈՒՆ (STATE)
# ==========================================
st.set_page_config(page_title="Solar Forecast Pro", page_icon="☀️", layout="wide")

# Ստեղծում ենք հիշողություն, որպեսզի Tab-երը փոխելիս տվյալները չկորչեն
if 'df_future' not in st.session_state:
    st.session_state.df_future = None

st.title("☀️ Արևային Էներգիայի Կանխատեսման և Ուսուցման MLOps Հարթակ")

# ==========================================
# 2. ՖՈՒՆԿՑԻԱՆԵՐ
# ==========================================
@st.cache_resource
def load_models():
    try:
        temp_model = joblib.load('temperature_model.pkl')
        power_model = joblib.load('power_model.pkl')
        with open('config.json', 'r') as f:
            config = json.load(f)
        return temp_model, power_model, config
    except:
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
        'IRRADIATION': [x / 1000.0 for x in hourly['shortwave_radiation']], # Վատտը դարձնում ենք կՎտ
        'T2M': hourly['temperature_2m'],
        'WS10M': hourly['wind_speed_10m'],
        'RH2M': hourly['relative_humidity_2m'],
        'PRECTOTCORR': hourly['precipitation']
    })
    return df

def fetch_nasa_history(lat, lon, start_date, end_date):
    url = (f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
           f"parameters=ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,WS10M,RH2M,PRECTOTCORR"
           f"&community=RE&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON")
    resp = requests.get(url).json()
    df = pd.DataFrame(resp['properties']['parameter']).reset_index().rename(columns={'index': 'DATE_TIME'})
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%Y%m%d%H')
    df.replace(-999.0, pd.NA, inplace=True)
    df.ffill(inplace=True)
    return df

# ==========================================
# 3. ՆԵՐԴԻՐՆԵՐԻ (TABS) ՍՏԵՂԾՈՒՄ
# ==========================================
temp_model, power_model, config = load_models()

tab1, tab2 = st.tabs(["📊 Կանխատեսում (Dashboard)", "🧠 Մոդելների Ուսուցում (Training)"])

# ==========================================
# ՆԵՐԴԻՐ 1: ԿԱՆԽԱՏԵՍՈՒՄ (DASHBOARD)
# ==========================================
with tab1:
    if power_model is None:
        st.warning("⚠️ Մոդելները դեռ ուսուցանված չեն: Խնդրում ենք անցնել «Մոդելների Ուսուցում» ներդիր։")
    else:
        st.markdown("### ⚙️ Կառավարման Վահանակ")
        
        # Դաշտերը դնում ենք գեղեցիկ 4 սյունակների մեջ
        col_lat, col_lon, col_days, col_btn = st.columns([1, 1, 2, 1])
        
        with col_lat:
            lat = st.number_input("Լայնություն (Lat)", value=float(config['location']['latitude']), format="%.4f")
        with col_lon:
            lon = st.number_input("Երկայնություն (Lon)", value=float(config['location']['longitude']), format="%.4f")
        with col_days:
            forecast_days = st.slider("Կանխատեսման հորիզոն (Օրեր)", min_value=1, max_value=7, value=3)
        with col_btn:
            st.write("") # Դատարկ տարածք կոճակը հավասարեցնելու համար
            run_forecast = st.button("🚀 Սկսել Կանխատեսումը", use_container_width=True, type="primary")

        # Եթե սեղմել են կոճակը, կատարում ենք հաշվարկ և ՊԱՀՊԱՆՈՒՄ հիշողության մեջ
        if run_forecast:
            with st.spinner('Միանում ենք օդերևութաբանական կայանին և կատարում հաշվարկներ...'):
                df_fut = fetch_future_weather(lat, lon, forecast_days)
                
                X_weather = df_fut[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
                pred_temp = temp_model.predict(X_weather)
                
                df_fut['MODULE_TEMPERATURE'] = pred_temp
                X_full = X_weather.copy()
                X_full.insert(1, 'MODULE_TEMPERATURE', pred_temp)
                df_fut['PREDICTED_DC_POWER'] = np.maximum(0, power_model.predict(X_full))
                
                # Պահպանում ենք սեսիայի մեջ
                st.session_state.df_future = df_fut

        # Եթե հիշողության մեջ կա պատրաստի հաշվարկ, նկարում ենք այն 
        # (Այս հատվածը կաշխատի նույնիսկ եթե մենք գնանք 2-րդ Tab ու հետ գանք)
        if st.session_state.df_future is not None:
            df_display = st.session_state.df_future
            
            st.markdown("---")
            with st.expander("🌤️ Դիտել ներբեռնված եղանակային տվյալները (Աղբյուր՝ Open-Meteo)"):
                st.dataframe(df_display, width='stretch')
            
            st.subheader("📊 Հիմնական Ցուցանիշներ")
            c1, c2, c3 = st.columns(3)
            c1.metric("Պիկային Հզորություն", f"{df_display['PREDICTED_DC_POWER'].max():,.0f} Վտ")
            c2.metric("Ընդհանուր Էներգիա", f"{(df_display['PREDICTED_DC_POWER'].sum() / 1000):,.1f} կՎտժ")
            c3.metric("Ժամանակահատված", f"{len(df_display)} ժամ")

            st.subheader("📈 Գեներացիայի Ինտերակտիվ Գրաֆիկ")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_display['DATE_TIME'], y=df_display['PREDICTED_DC_POWER'],
                                     mode='lines+markers', name='Հզորություն (Վտ)',
                                     line=dict(color='#00b894', width=3), fill='tozeroy', fillcolor='rgba(0, 184, 148, 0.2)'))
            fig.update_layout(xaxis_title="Ամսաթիվ և Ժամ", yaxis_title="DC Power (Վտ)", hovermode="x unified", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Վերջնական Աղյուսակ")
            st.dataframe(df_display[['DATE_TIME', 'MODULE_TEMPERATURE', 'PREDICTED_DC_POWER']], width='stretch')


# ==========================================
# ՆԵՐԴԻՐ 2: ՄՈԴԵԼՆԵՐԻ ՈՒՍՈՒՑՈՒՄ (TRAINING)
# ==========================================
with tab2:
    st.markdown("### 🧠 Արհեստական Ինտելեկտի Վերապատրաստում (Fine-Tuning)")
    st.markdown("Բեռնեք նոր կայանի տվյալները՝ մոդելները հատուկ այդ տեղանքի համար հարմարեցնելու նպատակով:")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        gen_file = st.file_uploader("1. Գեներացիայի բազա (CSV)", type=['csv'])
        train_lat = st.number_input("Նոր Կայանի Լայնություն (Lat)", value=40.18, format="%.4f")
    with col_t2:
        wea_file = st.file_uploader("2. Սենսորների բազա (CSV)", type=['csv'])
        train_lon = st.number_input("Նոր Կայանի Երկայնություն (Lon)", value=44.51, format="%.4f")

    if st.button("⚙️ Սկսել Տվյալների Մշակումը և Ուսուցումը", type="primary"):
        if gen_file and wea_file:
            try:
                st.info("1/4. Կարդում ենք բեռնված տվյալները...")
                df_gen = pd.read_csv(gen_file)
                df_wea = pd.read_csv(wea_file)
                
                df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'])
                df_wea['DATE_TIME'] = pd.to_datetime(df_wea['DATE_TIME'])
                
                st.info("2/4. Ներբեռնում ենք կլիմայական պատմությունը NASA POWER սերվերներից...")
                start_str = df_gen['DATE_TIME'].min().strftime('%Y%m%d')
                end_str = df_gen['DATE_TIME'].max().strftime('%Y%m%d')
                
                df_nasa = fetch_nasa_history(train_lat, train_lon, start_str, end_str)
                
                with st.expander("👀 Դիտել 3 բազաները նախքան միավորումը"):
                    st.write("**Գեներացիայի տվյալներ:**", df_gen)
                    st.write("**Սենսորների տվյալներ:**", df_wea)
                    st.write("**NASA արբանյակային տվյալներ:**", df_nasa)
                
                st.info("3/4. Միավորում ենք բազաները և մաքրում տվյալները...")
                df_local = pd.merge(df_gen, df_wea, on=['DATE_TIME', 'PLANT_ID'], suffixes=('_gen', '_wea'))
                df_local['HOUR_ROUNDED'] = df_local['DATE_TIME'].dt.floor('h')
                df_final = pd.merge(df_local, df_nasa, left_on='HOUR_ROUNDED', right_on='DATE_TIME', suffixes=('', '_nasa'))
                
                st.info("4/4. Սկսվում է XGBoost մոդելների ուսուցումը (կարող է տևել մի քանի րոպե)...")
                X_weather = df_final[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
                y_temp = df_final['MODULE_TEMPERATURE']
                
                X_full = df_final[['IRRADIATION', 'MODULE_TEMPERATURE', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
                y_power = df_final['DC_POWER']
                
                new_temp_model = XGBRegressor(learning_rate=0.1, max_depth=7, n_estimators=300, random_state=42)
                new_temp_model.fit(X_weather, y_temp)
                
                new_power_model = XGBRegressor(learning_rate=0.1, max_depth=7, n_estimators=300, random_state=42)
                new_power_model.fit(X_full, y_power)
                
                # Պահպանում
                joblib.dump(new_temp_model, 'temperature_model.pkl')
                joblib.dump(new_power_model, 'power_model.pkl')
                
                # Թարմացնել config-ը
                config_data = {"location": {"latitude": train_lat, "longitude": train_lon}}
                with open('config.json', 'w') as f:
                    json.dump(config_data, f)
                
                # Մաքրում ենք քեշը, որպեսզի հաջորդ անգամ "Կանխատեսում" ներդիրը կարդա ՆՈՐ մոդելները
                st.cache_resource.clear()
                
                st.success("✅ ՀԱՋՈՂՈՒԹՅՈՒՆ։ Մոդելները սովորեցին նոր տվյալներով և պատրաստ են օգտագործման (Անցեք 1-ին ներդիր):")
                st.balloons()
            
            except Exception as e:
                st.error(f"❌ Սխալ տեղի ունեցավ: Ստուգեք բեռնված CSV ֆայլերի կառուցվածքը: Տեխնիկական սխալ՝ {e}")
        else:
            st.warning("Խնդրում ենք բեռնել երկու CSV ֆայլերն էլ ուսուցումը սկսելու համար:")