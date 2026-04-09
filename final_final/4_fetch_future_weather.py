import json
import requests
import pandas as pd

print("--- ՔԱՅԼ 4: ԱՊԱԳԱ ԵՂԱՆԱԿԻ ԿԱՆԽԱՏԵՍՈՒՄ (OPEN-METEO) ---")
with open('config.json', 'r') as f:
    config = json.load(f)

LAT = config['location']['latitude']
LON = config['location']['longitude']
FUTURE_HOURS = config['future_forecast']['duration_hours']
output_file = config['future_forecast']['output_file']

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

df_fut.to_csv(output_file, index=False)
print(f"Ապագա {FUTURE_HOURS} ժամվա տվյալները պահպանվեցին '{output_file}'-ում:\n")