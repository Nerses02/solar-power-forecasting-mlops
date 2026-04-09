import requests
import pandas as pd
from datetime import datetime

print("Միանում ենք Open-Meteo օդերևութաբանական կայանին...")

# 1. Կարգավորում ենք պարամետրերը
# Հնդկաստանի արևային կայանի մոտավոր կոորդինատները (կարող ես փոխել Երևանի կոորդինատներով 40.18, 44.51)
LATITUDE = 19.99
LONGITUDE = 73.79

# API-ի հղումը և մեր ուզած պարամետրերը
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    # Խնդրում ենք ժամային տվյալներ 7 օրվա համար
    "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "shortwave_radiation"],
    "wind_speed_unit": "ms",  # Քամու արագությունը խնդրում ենք մետր/վայրկյանով (մեր մոդելին դա է պետք)
    "timezone": "auto"        # Ավտոմատ ճշտում է տեղական ժամային գոտին
}

# 2. Ուղարկում ենք հարցումը և ստանում տվյալները (JSON ֆորմատով)
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    hourly_data = data['hourly']
    
    # 3. Հավաքում ենք աղյուսակը ՃԻՇՏ մեր մոդելի սյունակների անվանումներով
    df_forecast = pd.DataFrame({
        'DATE_TIME': hourly_data['time'],
        'IRRADIATION': hourly_data['shortwave_radiation'],     # Արևի ճառագայթում (W/m²)
        'T2M': hourly_data['temperature_2m'],                  # Նույն օդի ջերմաստիճանը (NASA-ի պարամետրի անունով)
        'WS10M': hourly_data['wind_speed_10m'],                # Քամի (m/s)
        'RH2M': hourly_data['relative_humidity_2m'],           # Խոնավություն (%)
        'PRECTOTCORR': hourly_data['precipitation']            # Տեղումներ (mm)
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