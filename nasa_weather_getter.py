import requests
import pandas as pd

print("Միանում ենք Open-Meteo օդերևութաբանական կայանին...")

# 1. Կարգավորում ենք պարամետրերը
LATITUDE = 19.99
LONGITUDE = 73.79

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "shortwave_radiation"],
    "wind_speed_unit": "ms",
    "timezone": "auto"
}

# 2. Ուղարկում ենք հարցումը և ստանում տվյալները
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    hourly_data = data['hourly']
    
    # 3. Հավաքում ենք աղյուսակը ՃԻՇՏ մեր նոր մոդելի ձևաչափով (Առանց AMBIENT_TEMPERATURE-ի)
    df_forecast = pd.DataFrame({
        'DATE_TIME': hourly_data['time'],
        'IRRADIATION': hourly_data['shortwave_radiation'],     # Արևի ճառագայթում
        'T2M': hourly_data['temperature_2m'],                  # Օդի ջերմաստիճանը
        'WS10M': hourly_data['wind_speed_10m'],                # Քամի
        'RH2M': hourly_data['relative_humidity_2m'],           # Խոնավություն
        'PRECTOTCORR': hourly_data['precipitation']            # Տեղումներ
    })
    
    # Ամսաթիվը սարքում ենք DateTime օբյեկտ
    df_forecast['DATE_TIME'] = pd.to_datetime(df_forecast['DATE_TIME'])
    
    # 4. Պահպանում ենք նոր CSV ֆայլում
    filename = 'Future_Weather_Forecast.csv'
    df_forecast.to_csv(filename, index=False)
    
    print(f"\nՀԱՋՈՂՈՒԹՅՈՒՆ: Առաջիկա 7 օրվա ({len(df_forecast)} ժամ) կանխատեսումը բեռնված է:")
    print(f"Տվյալները պահպանվեցին '{filename}' ֆայլում:\n")
    print("Ահա առաջին 5 ժամերի տեսությունը.")
    print(df_forecast.head())

else:
    print(f"Սխալ տեղի ունեցավ API-ին միանալիս: Կոդ: {response.status_code}")