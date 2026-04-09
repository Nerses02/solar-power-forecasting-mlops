import json
import pandas as pd

print("--- ՔԱՅԼ 2: ՏՎՅԱԼՆԵՐԻ ՄԻԱՎՈՐՈՒՄ ԵՎ ՄԱՔՐՈՒՄ ---")
with open('config.json', 'r') as f:
    config = json.load(f)

df_gen = pd.read_csv(config['input_files']['generation_data'])
df_wea = pd.read_csv(config['input_files']['sensor_data'])
df_nasa = pd.read_csv(config['nasa_historical']['output_file'])

df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'], format='%d-%m-%Y %H:%M')
df_wea['DATE_TIME'] = pd.to_datetime(df_wea['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
df_nasa['DATE_TIME'] = pd.to_datetime(df_nasa['DATE_TIME'])

# Միացնում ենք լոկալները
df_local = pd.merge(df_gen, df_wea, on=['DATE_TIME', 'PLANT_ID'], suffixes=('_gen', '_wea'))
df_local['HOUR_ROUNDED'] = df_local['DATE_TIME'].dt.floor('h')

# Միացնում ենք NASA-ի հետ
df_final = pd.merge(df_local, df_nasa, left_on='HOUR_ROUNDED', right_on='DATE_TIME', suffixes=('', '_nasa'))

columns_to_keep = ['DATE_TIME', 'PLANT_ID', 'SOURCE_KEY_gen', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD', 
                   'SOURCE_KEY_wea', 'MODULE_TEMPERATURE', 'IRRADIATION', 'ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DNI', 
                   'ALLSKY_SFC_SW_DIFF', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']

df_final = df_final[columns_to_keep]
final_out = config['output_files']['final_merged_dataset']
df_final.to_csv(final_out, index=False)
print(f"Վերջնական բազան պատրաստ է ({len(df_final)} տող): Պահպանվեց '{final_out}' ֆայլում:\n")