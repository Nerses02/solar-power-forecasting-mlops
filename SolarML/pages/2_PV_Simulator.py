import streamlit as st
import pvlib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import fetch_nasa_annual, load_models_and_config

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
st.markdown("Համակարգը օգտագործում է Տիպիկ Օդերևութաբանական Տարվա (TMY) տվյալներ և արևի դիրքի աստղագիտական մաթեմատիկա:")

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
        
        # --- ՆՈՐ ԱՎԵԼԱՑՎԱԾ ՏՎՅԱԼՆԵՐ ԳՐԱՖԻԿՆԵՐԻ ՀԱՄԱՐ ---
        df['solar_elevation'] = solpos['apparent_elevation']
        df['solar_azimuth'] = solpos['azimuth']
        df['poa_global'] = poa['poa_global']
        df['temp_cell'] = temp_cell
        df['hour'] = df.index.hour
        df['date'] = df.index.date
        # -----------------------------------------------
        
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
    
    # Ստեղծում ենք ներդիրներ (Tabs)՝ էկրանը չծանրաբեռնելու համար
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
        st.markdown("Այս քարտեզը ցույց է տալիս կայանի աշխատանքը տարվա բոլոր օրերի և ժամերի ընթացքում:")
        
        # Պատրաստում ենք Pivot աղյուսակ Heatmap-ի համար
        pivot = df_res.pivot_table(values='AC_POWER_W', index='hour', columns='date', aggfunc='mean')
        
        fig2 = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index, 
            colorscale='YlOrRd', # Դեղինից-Կարմիր գույներ
            colorbar=dict(title="Վտ")
        ))
        fig2.update_layout(yaxis=dict(autorange="reversed", title="Օրվա ժամեր (0-23)"), xaxis_title="Ամսաթիվ")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Արևի Բարձրությունը և Հորիզոնի Ստվերարկումը (Ամառային Արևադարձ)")
        st.markdown("Գրաֆիկը ցույց է տալիս արևի դիրքը երկնքում 3 օրվա կտրվածքով: Կարմիր գծից ներքև գտնվող հատվածում կայանը չի աշխատում սարերի ստվերի պատճառով:")
        
        # Ընտրում ենք 3 պարզ օր հունիսին (Ամառային արևադարձին մոտ)
        df_summer = df_res[(df_res.index.month == 6) & (df_res.index.day >= 20) & (df_res.index.day <= 22)]
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_summer.index, y=df_summer['solar_elevation'], 
                                  mode='lines', name='Արևի Անկյուն (°)', line=dict(color='#f1c40f', width=3)))
        
        # Ավելացնում ենք Shading Cutoff գիծը
        cutoff = st.session_state.pv_settings['shading_cutoff']
        fig3.add_hline(y=cutoff, line_dash="dash", line_color="red", 
                       annotation_text=f"Ստվերի շեմ ({cutoff}°)", annotation_position="bottom right")
        
        fig3.update_layout(yaxis_title="Բարձրությունը հորիզոնից (°)", xaxis_title="Ժամանակ")
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.subheader("Ճառագայթման Վերլուծություն. POA ընդդեմ GHI")
        st.markdown(f"Ցույց է տալիս, թե ինչպես է վահանակի անկյունը (Tilt = {st.session_state.pv_settings['slope']}°) ավելացնում կլանվող էներգիան՝ հորիզոնական դիրքի համեմատ:")
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df_summer.index, y=df_summer['poa_global'], 
                                  mode='lines', fill='tozeroy', name='POA (Թեքված վահանակի վրա)', line=dict(color='#2ecc71')))
        fig4.add_trace(go.Scatter(x=df_summer.index, y=df_summer['ghi'], 
                                  mode='lines', name='GHI (Հորիզոնական)', line=dict(color='#95a5a6', dash='dot')))
        
        fig4.update_layout(yaxis_title="Ճառագայթում (Վտ/մ²)", xaxis_title="Ժամանակ")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("Բեռնել ամբողջական բազան")
    csv_data = df_res[['solar_elevation', 'solar_azimuth', 'poa_global', 'temp_cell', 'AC_POWER_W']].to_csv().encode('utf-8')
    st.download_button("📥 Ներբեռնել CSV (8760 ժամ)", data=csv_data, file_name="Annual_Simulation_Details.csv", mime="text/csv")