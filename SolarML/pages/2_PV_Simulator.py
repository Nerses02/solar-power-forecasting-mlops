import streamlit as st
import pvlib
import pandas as pd
import numpy as np
import requests
import datetime
import plotly.graph_objects as go
from utils import fetch_nasa_annual, load_models_and_config

# ==========================================
# 0. ՕԺԱՆԴԱԿ ՖՈՒՆԿՑԻԱ (PVGIS ՌԵԼԻԵՖ)
# ==========================================
@st.cache_data
def fetch_pvgis_horizon(lat, lon):
    """
    Քաշում է տեղանքի 360° հորիզոնի պրոֆիլը PVGIS արբանյակային տվյալներից:
    """
    # ՈՒՂՂՈՒՄ 1. Ճիշտ API հասցեն 'printhorizon' է
    url = f"https://re.jrc.ec.europa.eu/api/v5_2/printhorizon?lat={lat}&lon={lon}&outputformat=json"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if response.status_code != 200 or 'outputs' not in data:
            error_msg = data.get('message', 'Անհայտ սխալ սերվերից')
            st.warning(f"⚠️ PVGIS քարտեզը հասանելի չէ այս կոորդինատների համար ({error_msg}): Օգտագործվում է հարթ հորիզոն:")
            return [0, 360], [0, 0] 
            
        profile = data['outputs']['horizon_profile']
        
        parsed_data = []
        for p in profile:
            # ՈՒՂՂՈՒՄ 2. Ճիշտ բանալիները 'A' (Ազիմուտ) և 'H_hor' (Բարձրություն) են
            pvlib_azimuth = (p['A'] + 180) % 360
            parsed_data.append((pvlib_azimuth, p['H_hor']))
            
        parsed_data = sorted(parsed_data, key=lambda x: x[0])
        az_sorted = [d[0] for d in parsed_data]
        el_sorted = [d[1] for d in parsed_data]
        
        if az_sorted[0] != 0:
            az_sorted.insert(0, 0)
            el_sorted.insert(0, el_sorted[-1])
        if az_sorted[-1] != 360:
            az_sorted.append(360)
            el_sorted.append(el_sorted[0])
            
        return az_sorted, el_sorted
        
    except Exception as e:
        st.warning(f"⚠️ Ձախողվեց կապը PVGIS սերվերի հետ ({e}): Օգտագործվում է 0° հարթ հորիզոն:")
        return [0, 360], [0, 0]

# ==========================================
# 1. ԷՋԻ ԿԱՐԳԱՎՈՐՈՒՄՆԵՐ ԵՎ ՄՇՏԱԿԱՆ ՀԻՇՈՂՈՒԹՅՈՒՆ
# ==========================================
st.set_page_config(page_title="PV Simulator", page_icon="🏗️", layout="wide")

_, _, config = load_models_and_config()
default_lat = float(config['location']['latitude']) if config else 19.99
default_lon = float(config['location']['longitude']) if config else 73.79

if 'pv_settings' not in st.session_state:
    st.session_state.pv_settings = {
        'lat': default_lat,
        'lon': default_lon,
        'kwp': 10.0,
        'slope': 35,
        'azimuth': 180,
        'use_auto_horizon': True,
        'shading_cutoff': 15,
        'mounting': "Free-standing (Ազատ)",
        'loss_total': 14.0
    }

if 'sim_df' not in st.session_state:
    st.session_state.sim_df = None
if 'sim_monthly_yield' not in st.session_state:
    st.session_state.sim_monthly_yield = None
if 'sim_total_mwh' not in st.session_state:
    st.session_state.sim_total_mwh = None

st.title("🏗️ Տարեկան Ֆիզիկական Սիմուլյացիա (PV Simulator)")
st.markdown("Համակարգը օգտագործում է NASA TMY կլիման և **DVGIS ռելիեֆային քարտեզավորումը**:")

# ==========================================
# 2. ՁԱԽ ՎԱՀԱՆԱԿ: ՖԻԶԻԿԱԿԱՆ ՊԱՐԱՄԵՏՐԵՐ
# ==========================================
st.sidebar.header("Տեխնիկական Տվյալներ")

lat = st.sidebar.number_input("Լայնություն", value=st.session_state.pv_settings['lat'], format="%.6f", step=0.01)
st.session_state.pv_settings['lat'] = lat

lon = st.sidebar.number_input("Երկայնություն", value=st.session_state.pv_settings['lon'], format="%.6f", step=0.01)
st.session_state.pv_settings['lon'] = lon

kwp = st.sidebar.number_input("Դրվածքային Հզորություն (kWp)", value=st.session_state.pv_settings['kwp'], step=1.0)
st.session_state.pv_settings['kwp'] = kwp

# --- ԱՎՏՈ-ԿԱԼԻԲՐԱՑԻԱՅԻ ԲԼՈԿ (նախորդի պես) ---
st.sidebar.markdown("---")
with st.sidebar.expander("Ավտո-Կալիբրացիա (Օպտիմալացում)", expanded=False):
    theory_azimuth = 180 if lat >= 0 else 0
    theory_slope = max(0, int(abs(lat) - 5))
    st.info(f"💡 Սկսեք փնտրումը մոտավորապես {theory_slope}° թեքությունից և {theory_azimuth}° ազիմուտից:")
    complexity = st.selectbox("Փնտրման Բարդություն", ["Արագ (10° քայլով)", "Միջին (5° քայլով)", "Ճշգրիտ (2° քայլով)"])
    step_map = {"Արագ (10° քայլով)": 10, "Միջին (5° քայլով)": 5, "Ճշգրիտ (2° քայլով)": 2}
    step = step_map[complexity]
    
    if st.button("🔍 Գտնել Օպտիմալ Անկյունները", type="primary", use_container_width=True):
        with st.spinner("Կատարվում է կալիբրացիա..."):
            df_cal = fetch_nasa_annual(lat, lon, 2025)
            solpos_cal = pvlib.solarposition.get_solarposition(time=df_cal.index, latitude=lat, longitude=lon)
            # Հորիզոնի պարզեցված դիմակ կալիբրացիայի համար
            mask_cal = solpos_cal['apparent_elevation'] > st.session_state.pv_settings['shading_cutoff']
            dni_c, ghi_c, dhi_c = df_cal['dni'] * mask_cal, df_cal['ghi'] * mask_cal, df_cal['dhi'] * mask_cal
            u0_c, u1_c = (25.0, 6.84) if st.session_state.pv_settings['mounting'] == "Free-standing (Ազատ)" else (26.9, 6.76)
            
            slope_range = range(max(0, theory_slope - 20), min(90, theory_slope + 21), step)
            azimuth_range = range(theory_azimuth - 30, theory_azimuth + 31, step)
            
            max_yield, best_s, best_a = 0, theory_slope, theory_azimuth
            total_iters = len(slope_range) * len(azimuth_range)
            prog_bar, iters = st.progress(0), 0
            
            for s in slope_range:
                for a in azimuth_range:
                    poa_c = pvlib.irradiance.get_total_irradiance(
                        surface_tilt=s, surface_azimuth=a, solar_zenith=solpos_cal['apparent_zenith'], 
                        solar_azimuth=solpos_cal['azimuth'], dni=dni_c, ghi=ghi_c, dhi=dhi_c
                    )
                    temp_c = pvlib.temperature.faiman(poa_c['poa_global'], df_cal['temp_air'], df_cal['wind_speed'], u0_c, u1_c)
                    pdc_c = pvlib.pvsystem.pvwatts_dc(poa_c['poa_global'], temp_c, pdc0=kwp * 1000, gamma_pdc=-0.004)
                    if pdc_c.sum() > max_yield:
                        max_yield, best_s, best_a = pdc_c.sum(), s, a
                    iters += 1
                    prog_bar.progress(iters / total_iters)
            prog_bar.empty()
            st.session_state.pv_settings['slope'], st.session_state.pv_settings['azimuth'] = best_s, best_a
            st.success(f"✅ Գտնվել է! Թեքություն: {best_s}°, Ազիմուտ: {best_a}°")
            st.rerun()

# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Անկյուններ և Ռելիեֆ")
slope = st.sidebar.slider("Թեքություն (Slope/Tilt °)", 0, 90, st.session_state.pv_settings['slope'])
st.session_state.pv_settings['slope'] = slope

azimuth = st.sidebar.slider("Ազիմուտ (Azimuth °)", 0, 360, st.session_state.pv_settings['azimuth']) 
st.session_state.pv_settings['azimuth'] = azimuth

use_auto_horizon = st.sidebar.checkbox("🏔️ Ավտոմատ Ռելիեֆ", value=st.session_state.pv_settings['use_auto_horizon'], help="Ավտոմատ կբեռնի Ձեր շրջապատի սարերի ճշգրիտ բարձրությունը:")
st.session_state.pv_settings['use_auto_horizon'] = use_auto_horizon

if not use_auto_horizon:
    shading_cutoff = st.sidebar.slider("Ձեռքով Ստվերի Շեմ (°)", 0, 40, st.session_state.pv_settings['shading_cutoff'])
    st.session_state.pv_settings['shading_cutoff'] = shading_cutoff

st.sidebar.subheader("Մոնտաժ և Կորուստներ")
mounting_options = ["Free-standing (Ազատ)", "Roof-integrated (Տանիքին)"]
current_idx = mounting_options.index(st.session_state.pv_settings['mounting'])
mounting = st.sidebar.selectbox("Մոնտաժման Տիպ", mounting_options, index=current_idx)
st.session_state.pv_settings['mounting'] = mounting

loss_total = st.sidebar.number_input("Ընդհանուր Կորուստներ (%)", value=st.session_state.pv_settings['loss_total'])
st.session_state.pv_settings['loss_total'] = loss_total

run_sim = st.button("🚀 Սկսել Տարեկան Սիմուլյացիան", type="primary")

# ==========================================
# 3. ՀԻՄՆԱԿԱՆ ՀԱՇՎԱՐԿԸ (pvlib + PVGIS)
# ==========================================
if run_sim:
    with st.spinner("Կատարվում են աստղագիտական և ռելիեֆային հաշվարկներ..."):
        
        df = fetch_nasa_annual(lat, lon, 2025) 
        solpos = pvlib.solarposition.get_solarposition(time=df.index, latitude=lat, longitude=lon)
        
        # --- ՌԵԼԻԵՖԻ ՍՏՎԵՐԱՐԿՈՒՄԸ ---
        if use_auto_horizon:
            az_arr, el_arr = fetch_pvgis_horizon(lat, lon)
            # Համապատասխանեցնում ենք սարի բարձրությունը արևի յուրաքանչյուր ազիմուտին մաթեմատիկական ինտերպոլյացիայով
            actual_horizon = np.interp(solpos['azimuth'], az_arr, el_arr)
            mask = solpos['apparent_elevation'] > actual_horizon
            df['horizon_elevation'] = actual_horizon # Պահում ենք գրաֆիկի համար
        else:
            mask = solpos['apparent_elevation'] > shading_cutoff
            df['horizon_elevation'] = shading_cutoff
            
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=slope,
            surface_azimuth=azimuth,
            solar_zenith=solpos['apparent_zenith'],
            solar_azimuth=solpos['azimuth'],
            dni=df['dni'] * mask,
            ghi=df['ghi'] * mask,
            dhi=df['dhi'] * mask
        )
        
        u0, u1 = (25.0, 6.84) if mounting == "Free-standing (Ազատ)" else (26.9, 6.76)
        temp_cell = pvlib.temperature.faiman(poa['poa_global'], df['temp_air'], df['wind_speed'], u0, u1)
        pdc = pvlib.pvsystem.pvwatts_dc(poa['poa_global'], temp_cell, pdc0=kwp * 1000, gamma_pdc=-0.004)
        
        final_ac_power = pdc * (1 - (loss_total / 100.0))
        df['AC_POWER_W'] = final_ac_power.fillna(0) 
        
        df['solar_elevation'] = solpos['apparent_elevation']
        df['solar_azimuth'] = solpos['azimuth']
        df['poa_global'] = poa['poa_global']
        df['temp_cell'] = temp_cell
        df['hour'] = df.index.hour
        df['date'] = df.index.date
        
        total_mwh = df['AC_POWER_W'].sum() / 1_000_000
        df['Month'] = df.index.month
        monthly_yield = df.groupby('Month')['AC_POWER_W'].sum() / 1000 
        
        st.session_state.sim_df = df
        st.session_state.sim_monthly_yield = monthly_yield
        st.session_state.sim_total_mwh = total_mwh


# ==========================================
# 4. ՎԻԶՈՒԱԼԻԶԱՑԻԱ (ՆԵՐԴԻՐՆԵՐՈՎ)
# ==========================================
if st.session_state.sim_df is not None:
    df_res = st.session_state.sim_df
    
    st.success("✅ Սիմուլյացիան հաջողությամբ ավարտվեց:")
    st.metric("Տարեկան Ընդհանուր Գեներացիա", f"{st.session_state.sim_total_mwh:.2f} ՄՎտժ (MWh)")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Արտադրողականություն", 
        "🔥 Տարեկան Ջերմային Քարտեզ", 
        "☀️ Արևի Հետագիծ և Ստվեր", 
        "📈 Ճառագայթման Անկյուն"
    ])
    
    with tab1:
        st.subheader("Տարեկան Արտադրողականություն ըստ ամիսների (կՎտժ)")
        fig1 = go.Figure(data=[go.Bar(
            x=['Հնվ', 'Փտր', 'Մրտ', 'Ապր', 'Մյս', 'Հնս', 'Հլս', 'Օգս', 'Սպտ', 'Հկտ', 'Նյմ', 'Դկտ'], 
            y=st.session_state.sim_monthly_yield, 
            marker_color='#0984e3'
        )])
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.subheader("Արտադրողականության Ջերմային Քարտեզ (8760 ժամ)")
        pivot = df_res.pivot_table(values='AC_POWER_W', index='hour', columns='date', aggfunc='mean')
        fig2 = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale='YlOrRd'))
        fig2.update_layout(yaxis=dict(autorange="reversed", title="Օրվա ժամեր (0-23)"), xaxis_title="Ամսաթիվ")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Արևի Բարձրությունը և Հորիզոնի Ստվերարկումը")
        st.markdown("Մոխրագույն սիլուետը ցույց է տալիս տեղանքի իրական սարերի պրոֆիլը: Երբ արևը գտնվում է այդ սիլուետի մեջ, կայանը ստվերված է:")
        
        min_date, max_date = df_res.index.date.min(), df_res.index.date.max()
        date_range_3 = st.date_input("📅 Ընտրեք ժամանակահատվածը", value=(datetime.date(2025, 6, 20), datetime.date(2025, 6, 22)), min_value=min_date, max_value=max_date, key="date_tab3")
        
        if len(date_range_3) == 2:
            start_date, end_date = date_range_3
            df_plot3 = df_res[(df_res.index.date >= start_date) & (df_res.index.date <= end_date)]
            
            fig3 = go.Figure()
            # Արևի դիրքը (Դեղին գիծ)
            fig3.add_trace(go.Scatter(x=df_plot3.index, y=df_plot3['solar_elevation'], mode='lines', name='Արևի Անկյուն (°)', line=dict(color='#f1c40f', width=3)))
            
            # ՍԱՐԵՐԻ ՊՐՈՖԻԼԸ (Մոխրագույն լցվածք)
            fig3.add_trace(go.Scatter(
                x=df_plot3.index, y=df_plot3['horizon_elevation'], 
                mode='lines', fill='tozeroy', name='Սարերի պրոֆիլ (Ռելիեֆ)', 
                line=dict(color='#7f8c8d', width=2), fillcolor='rgba(127, 140, 141, 0.4)'
            ))
            
            fig3.update_layout(yaxis_title="Բարձրությունը հորիզոնից (°)", xaxis_title="Ժամանակ", hovermode="x unified")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Ընտրեք ավարտի ամսաթիվը:")

    with tab4:
        st.subheader("Ճառագայթման Վերլուծություն. POA ընդդեմ GHI")
        date_range_4 = st.date_input("📅 Ընտրեք ժամանակահատվածը", value=(datetime.date(2025, 6, 20), datetime.date(2025, 6, 22)), min_value=min_date, max_value=max_date, key="date_tab4")
        if len(date_range_4) == 2:
            start_date, end_date = date_range_4
            df_plot4 = df_res[(df_res.index.date >= start_date) & (df_res.index.date <= end_date)]
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=df_plot4.index, y=df_plot4['poa_global'], mode='lines', fill='tozeroy', name='POA (Թեքված վահանակ)', line=dict(color='#2ecc71')))
            fig4.add_trace(go.Scatter(x=df_plot4.index, y=df_plot4['ghi'], mode='lines', name='GHI (Հորիզոնական)', line=dict(color='#95a5a6', dash='dot')))
            fig4.update_layout(yaxis_title="Ճառագայթում (Վտ/մ²)", xaxis_title="Ժամանակ", hovermode="x unified")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("Ընտրեք ավարտի ամսաթիվը:")

    st.markdown("---")
    st.subheader("Բեռնել ամբողջական բազան")
    csv_data = df_res[['solar_elevation', 'horizon_elevation', 'solar_azimuth', 'poa_global', 'temp_cell', 'AC_POWER_W']].to_csv().encode('utf-8')
    st.download_button("📥 Ներբեռնել CSV (8760 ժամ)", data=csv_data, file_name="Annual_Simulation_Details.csv", mime="text/csv")