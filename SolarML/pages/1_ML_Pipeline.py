import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import json
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. ԷՋԻ ԿԱՐԳԱՎՈՐՈՒՄՆԵՐ ԵՎ ՀԻՇՈՂՈՒԹՅՈՒՆ (STATE)
# ==========================================
# Նշում. Multi-page հավելվածներում յուրաքանչյուր էջ կարող է ունենալ իր config-ը
st.set_page_config(page_title="Solar Forecast - ML Pipeline", page_icon="📈", layout="wide")

# Սեսիայի հիշողության սկզբնավորում, որպեսզի Tab-երը փոխելիս տվյալները չկորչեն
if 'df_future' not in st.session_state:
    st.session_state.df_future = None

st.title("📈 Արևային Էներգիայի Կանխատեսման և Ուսուցման MLOps Հարթակ")

# ==========================================
# 2. ՕԺԱՆԴԱԿ ՖՈՒՆԿՑԻԱՆԵՐ
# ==========================================

@st.cache_resource
def load_models():
    """Բեռնում է պահպանված մոդելները և config ֆայլը"""
    try:
        temp_model = joblib.load('./.env/temperature_model.pkl')
        power_model = joblib.load('./.env/power_model.pkl')
        with open('./.env/config.json', 'r') as f:
            config = json.load(f)
        return temp_model, power_model, config
    except:
        return None, None, None

def fetch_future_weather(lat, lon, days):
    """Բեռնում է ապագա եղանակը Open-Meteo-ից"""
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
        'IRRADIATION': [x / 1000.0 for x in hourly['shortwave_radiation']], # Վատտը -> կՎտ
        'T2M': hourly['temperature_2m'],
        'WS10M': hourly['wind_speed_10m'],
        'RH2M': hourly['relative_humidity_2m'],
        'PRECTOTCORR': hourly['precipitation']
    })
    return df

def fetch_nasa_history(lat, lon, start_date, end_date):
    """Բեռնում է պատմական կլիմայական տվյալները NASA POWER-ից"""
    url = (f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
           f"parameters=ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,WS10M,RH2M,PRECTOTCORR"
           f"&community=RE&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON")
    resp = requests.get(url).json()
    df = pd.DataFrame(resp['properties']['parameter']).reset_index().rename(columns={'index': 'DATE_TIME'})
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%Y%m%d%H')
    df.replace(-999.0, pd.NA, inplace=True)
    df.ffill(inplace=True)
    return df

# Մոդելների բեռնում
temp_model, power_model, config = load_models()

# ==========================================
# 3. ՆԵՐԴԻՐՆԵՐԻ (TABS) ՍՏԵՂԾՈՒՄ
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Կանխատեսում (Dashboard)", "🧠 Մոդելների Ուսուցում (Training)", "💡 Մոդելի Մեկնաբանելիություն (SHAP)"])

# ==========================================
# ՆԵՐԴԻՐ 1: ԿԱՆԽԱՏԵՍՈՒՄ (DASHBOARD)
# ==========================================
with tab1:
    if power_model is None:
        st.warning("⚠️ Մոդելները դեռ ուսուցանված չեն: Խնդրում ենք անցնել «Մոդելների Ուսուցում» ներդիր՝ նախագիծը սկզբնավորելու համար:")
    else:
        st.markdown("### ⚙️ Օպերատիվ Կառավարման Վահանակ")
        
        # Դաշտերի դասավորում սյունակներով
        col_lat, col_lon, col_days, col_btn = st.columns([1, 1, 2, 1])
        
        with col_lat:
            lat = st.number_input("Լայնություն (Lat)", value=float(config['location']['latitude']), format="%.4f")
        with col_lon:
            lon = st.number_input("Երկայնություն (Lon)", value=float(config['location']['longitude']), format="%.4f")
        with col_days:
            forecast_days = st.slider("Կանխատեսման հորիզոն (Օրեր)", min_value=1, max_value=7, value=3)
        with col_btn:
            st.write("") # Ուղղահայաց հավասարեցում
            run_forecast = st.button("🚀 Կանխատեսել", use_container_width=True, type="primary")

        # Հաշվարկային տրամաբանություն
        if run_forecast:
            with st.spinner('Միանում ենք եղանակային սերվերներին...'):
                df_fut = fetch_future_weather(lat, lon, forecast_days)
                
                # Կասկադային կանխատեսում
                X_weather = df_fut[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
                pred_temp = temp_model.predict(X_weather)
                
                df_fut['MODULE_TEMPERATURE'] = pred_temp
                X_full = X_weather.copy()
                X_full.insert(1, 'MODULE_TEMPERATURE', pred_temp)
                df_fut['PREDICTED_DC_POWER'] = np.maximum(0, power_model.predict(X_full))
                
                # Արդյունքի պահպանում սեսիայի մեջ
                st.session_state.df_future = df_fut

        # Արդյունքների վիզուալիզացիա (եթե առկա են հիշողության մեջ)
        if st.session_state.df_future is not None:
            df_display = st.session_state.df_future
            
            st.markdown("---")
            with st.expander("🌤️ Դիտել ներբեռնված եղանակային տվյալները"):
                st.dataframe(df_display, width='stretch')
            
            st.subheader("📊 Հիմնական Կանխատեսվող Ցուցանիշներ")
            c1, c2, c3 = st.columns(3)
            c1.metric("Պիկային Հզորություն", f"{df_display['PREDICTED_DC_POWER'].max():,.0f} Վտ")
            c2.metric("Ընդհանուր Սպասվող Էներգիա", f"{(df_display['PREDICTED_DC_POWER'].sum() / 1000):,.1f} կՎտժ")
            c3.metric("Ժամանակահատված", f"{len(df_display)} ժամ")

            st.subheader("📈 Գեներացիայի Ինտերակտիվ Գրաֆիկ")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_display['DATE_TIME'], 
                y=df_display['PREDICTED_DC_POWER'],
                mode='lines+markers', 
                name='Հզորություն (Վտ)',
                line=dict(color='#00b894', width=3), 
                fill='tozeroy', 
                fillcolor='rgba(0, 184, 148, 0.2)'
            ))
            fig.update_layout(xaxis_title="Ամսաթիվ և Ժամ", yaxis_title="DC Power (Վտ)", hovermode="x unified", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Մանրամասն Աղյուսակ")
            st.dataframe(df_display[['DATE_TIME', 'MODULE_TEMPERATURE', 'PREDICTED_DC_POWER']], width='stretch')


# ==========================================
# ՆԵՐԴԻՐ 2: ՄՈԴԵԼՆԵՐԻ ՈՒՍՈՒՑՈՒՄ (TRAINING)
# ==========================================
with tab2:
    st.markdown("### 🧠 Արհեստական Ինտելեկտի Վերապատրաստում (Fine-Tuning)")
    st.info("Այստեղ դուք կարող եք թարմացնել մոդելները՝ օգտագործելով կայանի իրական պատմական տվյալները:")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        gen_file = st.file_uploader("1. Գեներացիայի բազա (Plant Generation CSV)", type=['csv'])
        train_lat = st.number_input("Կայանի Լայնություն (Training Lat)", value=40.18, format="%.4f")
    with col_t2:
        wea_file = st.file_uploader("2. Սենսորների բազա (Weather Sensor CSV)", type=['csv'])
        train_lon = st.number_input("Կայանի Երկայնություն (Training Lon)", value=44.51, format="%.4f")

    if st.button("⚙️ Սկսել Ուսուցման Գործընթացը", type="primary"):
        if gen_file and wea_file:
            try:
                st.info("1/4. Մշակվում են բեռնված ֆայլերը...")
                df_gen = pd.read_csv(gen_file)
                df_wea = pd.read_csv(wea_file)
                
                df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'])
                df_wea['DATE_TIME'] = pd.to_datetime(df_wea['DATE_TIME'])
                
                st.info("2/4. Հարցում է կատարվում NASA POWER արբանյակային տվյալներին...")
                start_str = df_gen['DATE_TIME'].min().strftime('%Y%m%d')
                end_str = df_gen['DATE_TIME'].max().strftime('%Y%m%d')
                
                df_nasa = fetch_nasa_history(train_lat, train_lon, start_str, end_str)
                
                with st.expander("👀 Դիտել սինքրոնիզացվող բազաները"):
                    st.write("**Գեներացիա:**", df_gen)
                    st.write("**Սենսորներ:**", df_wea)
                    st.write("**NASA:**", df_nasa)
                
                st.info("3/4. Տվյալների միավորում և նախապատրաստում...")
                # Միավորում ըստ ժամանակի և PLANT_ID-ի
                df_local = pd.merge(df_gen, df_wea, on=['DATE_TIME', 'PLANT_ID'], suffixes=('_gen', '_wea'))
                df_local['HOUR_ROUNDED'] = df_local['DATE_TIME'].dt.floor('h')
                df_final = pd.merge(df_local, df_nasa, left_on='HOUR_ROUNDED', right_on='DATE_TIME', suffixes=('', '_nasa'))
                # save df_final to a temporary CSV for debugging
                df_final.to_csv('./.env/df_final_debug.csv', index=False)
                
                st.info("4/4. XGBoost մոդելների վարժեցում... (սպասեք փուչիկներին 🎈)")
                
                # Մոդել 1. Ջերմաստիճան
                X_weather = df_final[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
                y_temp = df_final['MODULE_TEMPERATURE']
                new_temp_model = XGBRegressor(learning_rate=0.1, max_depth=7, n_estimators=300, random_state=42)
                new_temp_model.fit(X_weather, y_temp)
                
                # Մոդել 2. Հզորություն
                X_full = df_final[['IRRADIATION', 'MODULE_TEMPERATURE', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
                y_power = df_final['DC_POWER']
                new_power_model = XGBRegressor(learning_rate=0.1, max_depth=7, n_estimators=300, random_state=42)
                new_power_model.fit(X_full, y_power)
                
                # Արդյունքների պահպանում սկավառակի վրա
                joblib.dump(new_temp_model, './.env/temperature_model.pkl')
                joblib.dump(new_power_model, './.env/power_model.pkl')
                
                # Կոնֆիգի թարմացում
                new_config = {"location": {"latitude": train_lat, "longitude": train_lon}}
                with open('./.env/config.json', 'w') as f:
                    json.dump(new_config, f)
                
                # Քեշի մաքրում՝ նոր մոդելները ակտիվացնելու համար
                st.cache_resource.clear()
                
                st.success("✅ Մոդելները հաջողությամբ թարմացվել են և պատրաստ են աշխատանքի:")
                st.balloons()
            
            except Exception as e:
                st.error(f"❌ Տվյալների մշակման սխալ: Համոզվեք, որ CSV սյունակները համապատասխանում են ստանդարտին: {e}")
        else:
            st.warning("Ուսուցումը սկսելու համար անհրաժեշտ է բեռնել երկու ֆայլերն էլ:")


# ==========================================
# ՆԵՐԴԻՐ 3: ՄՈԴԵԼԻ ՄԵԿՆԱԲԱՆԵԼԻՈՒԹՅՈՒՆ (SHAP)
# ==========================================
with tab3:
    st.markdown("### 💡 Արհեստական Ինտելեկտի Բացատրելիություն (SHAP Analysis)")
    
    if power_model is None:
        st.warning("⚠️ Մոդելը բացակայում է: Խնդրում ենք նախ վարժեցնել այն «Մոդելների Ուսուցում» բաժնում:")
    elif st.session_state.df_future is None:
        st.info("👈 Խնդրում ենք նախ անցնել «Կանխատեսում» ներդիր, մուտքագրել օրերը և սեղմել «Կանխատեսել», որպեսզի համակարգն ունենա տվյալներ վերլուծության համար:")
    else:
        st.markdown("Այս բաժինը թույլ է տալիս հասկանալ «Սև արկղի» (XGBoost) կայացրած որոշումները։ Տեսեք, թե ինչպես են օդերևութաբանական գործոններն ազդել վերջնական հզորության վրա:")
        
        # Վերցնում ենք Կանխատեսման ներդիրում գեներացված ապագա տվյալները
        df_shap = st.session_state.df_future.copy()
        X_shap = df_shap[['IRRADIATION', 'MODULE_TEMPERATURE', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
        
        # SHAP արժեքների հաշվարկման ֆունկցիա (քեշավորված արագության համար)
        @st.cache_data
        def calculate_shap_values(_model, X):
            explainer = shap.TreeExplainer(_model)
            shap_values = explainer(X)
            
            armenian_names = {
                'IRRADIATION': 'Ճառագայթում (կՎտ/մ²)',
                'MODULE_TEMPERATURE': 'Վահանակի Ջերմ. (°C)',
                'T2M': 'Օդի Ջերմաստիճան (°C)',
                'WS10M': 'Քամու Արագություն (մ/վ)',
                'RH2M': 'Խոնավություն (%)',
                'PRECTOTCORR': 'Տեղումներ (մմ)'
            }
            shap_values.feature_names = [armenian_names.get(name, name) for name in X.columns]
            
            return explainer, shap_values

        with st.spinner("Հաշվարկվում են SHAP մատրիցաները..."):
            explainer, shap_values = calculate_shap_values(power_model, X_shap)
        
        # --- 1. ԳԼՈԲԱԼ ՄԵԿՆԱԲԱՆՈՒԹՅՈՒՆ (Summary Plot) ---
        st.markdown("---")
        st.subheader("1. Հատկանիշների գլոբալ ազդեցությունը (Summary Plot)")
        st.write("Այս գրաֆիկը ցույց է տալիս, թե ընդհանուր առմամբ որ պարամետրերն ունեն ամենամեծ ազդեցությունը մոդելի վրա։ Կարմիր կետերը ցույց են տալիս տվյալ պարամետրի բարձր արժեքը, իսկ կապույտը՝ ցածր։")
        
        fig_summary = plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values.values, X_shap, feature_names=shap_values.feature_names, plot_type="dot", show=False)
        plt.subplots_adjust(left=0.35) 
        st.pyplot(fig_summary)
        plt.clf()

        # --- 2. ԼՈԿԱԼ ՄԵԿՆԱԲԱՆՈՒԹՅՈՒՆ (Bar Plot) ---
        st.markdown("---")
        st.subheader("2. Լոկալ ազդեցություն (Մեկ կոնկրետ ժամվա վերլուծություն)")
        st.write("Այս գրաֆիկը ցույց է տալիս, թե տվյալ պահին որ գործոններն են առավելապես նպաստել կամ խոչընդոտել էներգիայի արտադրությանը:")
        
        time_options = df_shap['DATE_TIME'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        selected_time = st.selectbox("📅 Ընտրեք ամսաթիվը և ժամը:", time_options)
        hour_index = time_options.index(selected_time)
        
        fig_bar = plt.figure(figsize=(12, 6)) 
        
        shap.plots.bar(shap_values[hour_index], show=False)
        
        ax = plt.gca()
        ax.tick_params(axis='y', pad=40) 
        ax.grid(axis='x', color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
        
        # =================================================================
        # ՆՈՐ ԼՈՒԾՈՒՄ: Ձեռքով հաշվում և ավելացնում ենք բացակայող թվերը
        
        # 1. Վերցնում ենք բազային արժեքը (Expected Value)
        base_val = shap_values[hour_index].base_values
        
        # 2. Վերջնական կանխատեսումը հավասար է բազային արժեք + բոլոր ազդեցությունները
        final_val = base_val + shap_values[hour_index].values.sum()
        
        # 3. Գրում ենք ԲԱԶԱՅԻՆ արժեքը գրաֆիկի ՆԵՐՔԵՎՈՒՄ ԱՋԱԿՈՂՄՅԱՆ մասում
        plt.figtext(0.95, 0.02, rf"$E[f(X)]$ = {base_val:.3f}", 
                    fontsize=12, color='#777777', ha='right', va='bottom')
        
        # 4. Գրում ենք ՎԵՐՋՆԱԿԱՆ արժեքը գրաֆիկի ՎԵՐԵՎՈՒՄ ԱՋԱԿՈՂՄՅԱՆ մասում
        plt.figtext(0.95, 0.95, rf"$f(x)$ = {final_val:.3f}", 
                    fontsize=14, color='#222222', ha='right', va='top', weight='bold')
        # =================================================================
        
        # Մի փոքր ավելի մեծ տեղ ենք թողնում վերևից ու ներքևից մեր գրած թվերի համար
        plt.subplots_adjust(left=0.40, right=0.95, top=0.85, bottom=0.15) 
        
        st.pyplot(fig_bar)
        plt.clf()