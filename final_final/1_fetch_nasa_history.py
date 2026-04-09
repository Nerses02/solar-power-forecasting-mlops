import json
import requests
import pandas as pd

print("--- ՔԱՅԼ 1: ԲԵՌՆՎՈՒՄ ԵՆ ՊԱՏՄԱԿԱՆ ՏՎՅԱԼՆԵՐԸ NASA-ԻՑ ---")
with open('config.json', 'r') as f:
    config = json.load(f)

LAT = config['location']['latitude']
LON = config['location']['longitude']
y_start = config['nasa_historical']['start_year']
y_end = config['nasa_historical']['end_year']
output_file = config['nasa_historical']['output_file']

url = (f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
       f"parameters=ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,WS10M,RH2M,PRECTOTCORR"
       f"&community=RE&longitude={LON}&latitude={LAT}&start={y_start}0101&end={y_end}1231&format=JSON")

resp = requests.get(url).json()
df = pd.DataFrame(resp['properties']['parameter']).reset_index().rename(columns={'index': 'DATE_TIME'})
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%Y%m%d%H')
df.replace(-999.0, pd.NA, inplace=True)
df.ffill(inplace=True)

df.to_csv(output_file, index=False)
print(f"Հաջողությամբ պահպանվեց '{output_file}' ֆայլում:\n")