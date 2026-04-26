import streamlit as st
import pvlib
import pandas as pd
import plotly.graph_objects as go
from utils import fetch_nasa_annual, load_models_and_config

# ==========================================
# 1. ԷՋԻ ԿԱՐԳԱՎՈՐՈՒՄՆԵՐ ԵՎ ՄՇՏԱԿԱՆ ՀԻՇՈՂՈՒԹՅՈՒՆ
# ==========================================
st.set_page_config(page_title="PV Simulator", page_icon="🏗️", layout="wide")

_, _, config = load_models_and_config()
default_lat = float(config['location']['latitude']) if config else 40.18
default_lon = float(config['location']['longitude']) if config else 44.51

# Ստեղծում ենք Անխափան Հիշողություն (Persistent Dictionary) ՊԱՐԱՄԵՏՐԵՐԻ համար
if 'pv_settings' not in st.session_state:
    st.session_state.pv_settings = {
        'lat': default_lat,
        'lon': default_lon,
        'kwp': 10.0,
        'slope': 35,
        'azimuth': 180,
        'shading_cutoff': 15,
        'mounting': "Free-standing (Ազատ)",
        'loss_total': 14.0
    }

# Ստեղծում ենք Հիշողություն ԱՐԴՅՈՒՆՔՆԵՐԻ համար
if 'sim_df' not in st.session_state:
    st.session_state.sim_df = None
if 'sim_monthly_yield' not in st.session_state:
    st.session_state.sim_monthly_yield = None
if 'sim_total_mwh' not in st.session_state:
    st.session_state.sim_total_mwh = None

st.title("🏗️ Տարեկան Ֆիզիկական Սիմուլյացիա (PV Simulator)")
st.markdown("Համակարգը օգտագործում է Տիպիկ Օդերևութաբանական Տարվա (TMY) տվյալներ և արևի դիրքի աստղագիտական մաթեմատիկա:")

# ==========================================
# 2. ՁԱԽ ՎԱՀԱՆԱԿ: ՖԻԶԻԿԱԿԱՆ ՊԱՐԱՄԵՏՐԵՐ
# ==========================================
st.sidebar.header("Տեխնիկական Տվյալներ")

# Կարդում ենք value-ն մեր բառարանից և անմիջապես թարմացնում այն՝ եթե օգտատերը փոխի թիվը
lat = st.sidebar.number_input("Լայնություն", value=st.session_state.pv_settings['lat'], format="%.6f", step=0.01)
st.session_state.pv_settings['lat'] = lat

lon = st.sidebar.number_input("Երկայնություն", value=st.session_state.pv_settings['lon'], format="%.6f", step=0.01)
st.session_state.pv_settings['lon'] = lon

kwp = st.sidebar.number_input("Դրվածքային Հզորություն (kWp)", value=st.session_state.pv_settings['kwp'], step=1.0)
st.session_state.pv_settings['kwp'] = kwp

st.sidebar.subheader("Անկյուններ և Ռելիեֆ")
slope = st.sidebar.slider("Թեքություն (Slope/Tilt °)", 0, 90, st.session_state.pv_settings['slope'])
st.session_state.pv_settings['slope'] = slope

azimuth = st.sidebar.slider("Ազիմուտ (Azimuth °)", 0, 360, st.session_state.pv_settings['azimuth']) 
st.session_state.pv_settings['azimuth'] = azimuth

shading_cutoff = st.sidebar.slider("Հորիզոնի Ստվեր (Terrain Cutoff °)", 0, 40, st.session_state.pv_settings['shading_cutoff'], help="Արևի նվազագույն անկյունը, որից ներքև սարերը ծածկում են արևը:")
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
# 3. ՀԻՄՆԱԿԱՆ ՀԱՇՎԱՐԿԸ (pvlib)
# ==========================================
if run_sim:
    with st.spinner("Քաշում ենք տարեկան NASA TMY տվյալները և հաշվում աստղագիտական արդյունքները..."):
        
        df = fetch_nasa_annual(lat, lon, 2024) 
        
        solpos = pvlib.solarposition.get_solarposition(time=df.index, latitude=lat, longitude=lon)
        
        mask = solpos['apparent_elevation'] > shading_cutoff
        
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
        final_ac_power = final_ac_power.fillna(0) 
        
        df['AC_POWER_W'] = final_ac_power
        
        # Պահպանում ենք արդյունքները սեսիայի մեջ
        total_mwh = df['AC_POWER_W'].sum() / 1_000_000
        df['Month'] = df.index.month
        monthly_yield = df.groupby('Month')['AC_POWER_W'].sum() / 1000 
        
        st.session_state.sim_df = df
        st.session_state.sim_monthly_yield = monthly_yield
        st.session_state.sim_total_mwh = total_mwh


# ==========================================
# 4. ՎԻԶՈՒԱԼԻԶԱՑԻԱ (Կարդում ենք հիշողությունից)
# ==========================================
if st.session_state.sim_df is not None:
    st.success("✅ Սիմուլյացիան հաջողությամբ ավարտվեց:")
    
    st.metric("Տարեկան Ընդհանուր Գեներացիա", f"{st.session_state.sim_total_mwh:.2f} ՄՎտժ (MWh)")
    
    st.subheader("Տարեկան Արտադրողականություն ըստ ամիսների (կՎտժ)")
    fig = go.Figure(data=[go.Bar(
        x=['Հնվ', 'Փտր', 'Մրտ', 'Ապր', 'Մյս', 'Հնս', 'Հլս', 'Օգս', 'Սպտ', 'Հկտ', 'Նյմ', 'Դկտ'], 
        y=st.session_state.sim_monthly_yield, 
        marker_color='#0984e3'
    )])
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Բեռնել 8760 ժամվա հաշվետվությունը")
    csv_data = st.session_state.sim_df[['AC_POWER_W', 'temp_air']].to_csv().encode('utf-8')
    st.download_button("📥 Ներբեռնել CSV", data=csv_data, file_name="Annual_Simulation.csv", mime="text/csv")