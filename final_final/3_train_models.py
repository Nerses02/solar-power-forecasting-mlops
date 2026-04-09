import json
import pandas as pd
import joblib
from xgboost import XGBRegressor

print("--- ՔԱՅԼ 3: ԱՐՀԵՍՏԱԿԱՆ ԻՆՏԵԼԵԿՏԻ ՈՒՍՈՒՑՈՒՄ ---")
with open('config.json', 'r') as f:
    config = json.load(f)

df_final = pd.read_csv(config['output_files']['final_merged_dataset'])

# Նախապատրաստում ենք տվյալները
X_weather = df_final[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
y_temp = df_final['MODULE_TEMPERATURE']

X_full = df_final[['IRRADIATION', 'MODULE_TEMPERATURE', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
y_power = df_final['DC_POWER']

print("Ուսուցանվում է Ջերմաստիճանի Մոդելը...")
temp_model = XGBRegressor(learning_rate=0.1, max_depth=6, n_estimators=250, random_state=42)
temp_model.fit(X_weather, y_temp)

print("Ուսուցանվում է Հզորության Գլխավոր Մոդելը...")
power_model = XGBRegressor(learning_rate=0.1, max_depth=7, n_estimators=300, random_state=42)
power_model.fit(X_full, y_power)

# Պահպանում ենք մոդելները (Սա ամենակարևոր նորամուծությունն է)
joblib.dump(temp_model, 'temperature_model.pkl')
joblib.dump(power_model, 'power_model.pkl')
print("Մոդելները հաջողությամբ պահպանվեցին որպես .pkl ֆայլեր:\n")