import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. ԿԱՐԴՈՒՄ ԵՆՔ CONFIG ՖԱՅԼԸ
# ==========================================
print("--- ՍԿՍՎՈՒՄ Է ԱՎՏՈՄԱՏԱՑՎԱԾ ԽՈՂՈՎԱԿԱՇԱՐԸ (PIPELINE) ---")
with open('config.json', 'r') as f:
    config = json.load(f)

LAT = config['location']['latitude']
LON = config['location']['longitude']
FUTURE_HOURS = config['future_forecast']['duration_hours']

# ==========================================
# 2. ԵՆԹԱԾՐԱԳՐԵՐ (Sub-scripts / Fetchers)
# ==========================================
def run_nasa_weather_getter():
    print("-> Բեռնվում են պատմական տվյալները NASA-ից...")
    y_start, y_end = config['nasa_historical']['start_year'], config['nasa_historical']['end_year']
    url = (f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
           f"parameters=ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,WS10M,RH2M,PRECTOTCORR"
           f"&community=RE&longitude={LON}&latitude={LAT}&start={y_start}0101&end={y_end}1231&format=JSON")
    resp = requests.get(url).json()
    df = pd.DataFrame(resp['properties']['parameter']).reset_index().rename(columns={'index': 'DATE_TIME'})
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%Y%m%d%H')
    df.replace(-999.0, pd.NA, inplace=True); df.ffill(inplace=True)
    df.to_csv(config['nasa_historical']['output_file'], index=False)
    return df

def run_predicted_future_weather():
    print(f"-> Բեռնվում են ապագա {FUTURE_HOURS} ժամվա եղանակային տվյալները Open-Meteo-ից...")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": LAT, "longitude": LON, 
              "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "shortwave_radiation"],
              "wind_speed_unit": "ms", "timezone": "auto", "forecast_hours": FUTURE_HOURS}
    hourly = requests.get(url, params=params).json()['hourly']
    df_fut = pd.DataFrame({
        'DATE_TIME': pd.to_datetime(hourly['time']),
        'IRRADIATION': hourly['shortwave_radiation'], 'T2M': hourly['temperature_2m'],
        'WS10M': hourly['wind_speed_10m'], 'RH2M': hourly['relative_humidity_2m'],
        'PRECTOTCORR': hourly['precipitation']
    })
    df_fut.to_csv(config['future_forecast']['output_file'], index=False)
    return df_fut

# ==========================================
# 3. ՏՎՅԱԼՆԵՐԻ ՄԻԱՎՈՐՈՒՄ (Data Merging)
# ==========================================
print("\n--- ՏՎՅԱԼՆԵՐԻ ՄԻԱՎՈՐՈՒՄ ԵՎ ՄԱՔՐՈՒՄ ---")
df_gen = pd.read_csv(config['input_files']['generation_data'])
df_wea = pd.read_csv(config['input_files']['sensor_data'])

# Ֆորմատավորում ենք ամսաթվերը
df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'], format='%d-%m-%Y %H:%M')
df_wea['DATE_TIME'] = pd.to_datetime(df_wea['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

# Միացնում ենք երկու Լոկալ ֆայլերը
df_local = pd.merge(df_gen, df_wea, on=['DATE_TIME', 'PLANT_ID'], suffixes=('_gen', '_wea'))

# Ստանում ենք NASA-ի տվյալները (կանչում ենք ֆունկցիան)
df_nasa = run_nasa_weather_getter()

# Միացնում ենք NASA-ի հետ (կլորացնելով ժամերը)
df_local['HOUR_ROUNDED'] = df_local['DATE_TIME'].dt.floor('h')
df_final = pd.merge(df_local, df_nasa, left_on='HOUR_ROUNDED', right_on='DATE_TIME', suffixes=('', '_nasa'))

# Մաքրում ենք և պահպանում վերջնական ստանդարտացված CSV-ն
columns_to_keep = ['DATE_TIME', 'PLANT_ID', 'SOURCE_KEY_gen', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD', 
                   'SOURCE_KEY_wea', 'MODULE_TEMPERATURE', 'IRRADIATION', 'ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DNI', 
                   'ALLSKY_SFC_SW_DIFF', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']
# NASA-ի ճառագայթումն ու լոկալ ճառագայթումը կարող են խառնվել, մենք պահում ենք լոկալը որպես հիմնական (IRRADIATION)
df_final = df_final[columns_to_keep]
df_final.to_csv(config['output_files']['final_merged_dataset'], index=False)
print(f"Վերջնական բազան ստեղծված է ({len(df_final)} տող):")

# ==========================================
# 4. ՄՈԴԵԼԻ ՈՒՍՈՒՑՈՒՄ (Cascaded ML)
# ==========================================
print("\n--- ԱՐՀԵՍՏԱԿԱՆ ԻՆՏԵԼԵԿՏԻ ՈՒՍՈՒՑՈՒՄ ---")
X_weather = df_final[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
y_temp = df_final['MODULE_TEMPERATURE']
X_full = df_final[['IRRADIATION', 'MODULE_TEMPERATURE', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
y_power = df_final['DC_POWER']

print("1. Ուսուցանվում է Ջերմաստիճանի Մոդելը...")
temp_model = XGBRegressor(learning_rate=0.1, max_depth=6, n_estimators=250, random_state=42).fit(X_weather, y_temp)

print("2. Ուսուցանվում է Հզորության Գլխավոր Մոդելը...")
power_model = XGBRegressor(learning_rate=0.1, max_depth=7, n_estimators=300, random_state=42).fit(X_full, y_power)

# ==========================================
# 5. ԱՊԱԳԱՅԻ ԿԱՆԽԱՏԵՍՈՒՄ
# ==========================================
print("\n--- ԱՊԱԳԱՅԻ ԿԱՆԽԱՏԵՍՈՒՄ ---")
df_future = run_predicted_future_weather()
X_future = df_future[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]

# Կասկադային գուշակություն
pred_temp = temp_model.predict(X_future)
X_future_full = X_future.copy()
X_future_full.insert(1, 'MODULE_TEMPERATURE', pred_temp)
future_power = power_model.predict(X_future_full)
df_future['PREDICTED_DC_POWER'] = np.maximum(0, future_power)

# ==========================================
# 6. ՊԱՏՄԱԿԱՆԻ ԵՎ ԱՊԱԳԱՅԻ ՎԻԶՈՒԱԼԻԶԱՑԻԱ
# ==========================================
print("\n--- ԳՐԱՖԻԿԻ ԿԱՌՈՒՑՈՒՄ ---")
# Վերցնում ենք 1 ինվերտոր վերջին 30 օրվա համար
inverter_id = df_final['SOURCE_KEY_gen'].unique()[0]
df_past = df_final[df_final['SOURCE_KEY_gen'] == inverter_id].sort_values('DATE_TIME').tail(720) # Վերջին 30 օրը (~720 ժամ)

# Որպեսզի գրաֆիկը անընդհատ լինի, ապագա ժամանակը կպցնում ենք պատմականի ավարտին (կարևոր է թեզի էսթետիկայի համար)
last_historical_date = df_past['DATE_TIME'].iloc[-1]
simulated_future_dates = [last_historical_date + pd.Timedelta(hours=i) for i in range(1, len(df_future) + 1)]

plt.figure(figsize=(16, 6))

# Նկարում ենք անցյալը (Կապույտ)
plt.plot(df_past['DATE_TIME'], df_past['DC_POWER'], label='Անցյալի Իրական Արտադրություն (Historical Actuals)', color='royalblue', linewidth=1.5)

# Նկարում ենք ապագան (Կանաչ)
plt.plot(simulated_future_dates, df_future['PREDICTED_DC_POWER'], label=f'Ապագա Կանխատեսում ({FUTURE_HOURS} ժամ)', color='mediumseagreen', linewidth=2, linestyle='-')

# Բաժանիչ ուղղահայաց գիծ
plt.axvline(x=last_historical_date, color='red', linestyle='--', linewidth=2, label='Ներկա Պահը (Now)')

plt.title(f'Արևային Էներգիայի Խորը Վերլուծություն և Կանխատեսում (Inverter: {inverter_id})', fontsize=14)
plt.xlabel('Ժամանակ', fontsize=12)
plt.ylabel('Հզորություն (DC Power)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.4)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('End_to_End_Forecast.png', dpi=300)
plt.show()

print("\n!!! ՀԱՄԱԿԱՐԳԸ ՀԱՋՈՂՈՒԹՅԱՄԲ ԱՎԱՐՏԵՑ ԻՐ ԱՇԽԱՏԱՆՔԸ !!!")