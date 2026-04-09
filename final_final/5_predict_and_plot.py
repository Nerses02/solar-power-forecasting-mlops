import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

print("--- ՔԱՅԼ 5: ՎԵՐՋՆԱԿԱՆ ԿԱՆԽԱՏԵՍՈՒՄ ԵՎ ՎԻԶՈՒԱԼԻԶԱՑԻԱ ---")

with open('config.json', 'r') as f:
    config = json.load(f)

FUTURE_HOURS = config['future_forecast']['duration_hours']

# 1. Բեռնում ենք ֆայլերը և ՄՈԴԵԼՆԵՐԸ
df_final = pd.read_csv(config['output_files']['final_merged_dataset'])
df_final['DATE_TIME'] = pd.to_datetime(df_final['DATE_TIME'])

df_future = pd.read_csv(config['future_forecast']['output_file'])
df_future['DATE_TIME'] = pd.to_datetime(df_future['DATE_TIME'])

print("Բեռնվում են ուսուցանված մոդելները...")
temp_model = joblib.load('temperature_model.pkl')
power_model = joblib.load('power_model.pkl')

# 2. Ապագայի Կանխատեսում
X_future = df_future[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
pred_temp = temp_model.predict(X_future)

X_future_full = X_future.copy()
X_future_full.insert(1, 'MODULE_TEMPERATURE', pred_temp)
df_future['PREDICTED_DC_POWER'] = np.maximum(0, power_model.predict(X_future_full))

# 3. Պատմական Տվյալների նախապատրաստում (Backtest)
inverter_id = df_final['SOURCE_KEY_gen'].unique()[0]
df_past = df_final[df_final['SOURCE_KEY_gen'] == inverter_id].sort_values('DATE_TIME').tail(720) 

eval_len = int(len(df_past) * 0.2)
df_eval = df_past.tail(eval_len).copy()

X_eval_weather = df_eval[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
eval_pred_temp = temp_model.predict(X_eval_weather)
X_eval_full = X_eval_weather.copy()
X_eval_full.insert(1, 'MODULE_TEMPERATURE', eval_pred_temp)
df_eval['BACKTEST_POWER'] = np.maximum(0, power_model.predict(X_eval_full))

last_historical_date = df_past['DATE_TIME'].iloc[-1]
simulated_future_dates = [last_historical_date + pd.Timedelta(hours=i) for i in range(1, len(df_future) + 1)]

print("Կառուցվում են Էկրանները...")

# ---------------------------------------------------------
# ԷՋ 1: ԱՆՑՅԱԼ ԵՎ BACKTEST
# ---------------------------------------------------------
plt.figure(figsize=(16, 6))
plt.plot(df_past['DATE_TIME'], df_past['DC_POWER'], label='Իրական Արտադրություն (Actuals)', color='royalblue', linewidth=1.5)
plt.plot(df_eval['DATE_TIME'], df_eval['BACKTEST_POWER'], label='Մոդելի Ստուգում (Backtest 20%)', color='darkorange', linewidth=2, linestyle='--')

# ՆԵՐԿԱ ՊԱՀԻ ԳԻԾԸ ՀԵՌԱՑՎԱԾ Է ԱՅՍՏԵՂԻՑ

plt.title(f'ՄԱՍ 1. Անցյալի Իրական Արտադրություն և Հետադարձ Ստուգում (Inverter: {inverter_id})', fontsize=14, fontweight='bold')
plt.ylabel('Հզորություն (DC Power)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.5, which='both')

ax1 = plt.gca()
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Part_1_Historical_Backtest.png', dpi=300)
plt.show()

# ---------------------------------------------------------
# ԷՋ 2: ԱՊԱԳԱ ԿԱՆԽԱՏԵՍՈՒՄ
# ---------------------------------------------------------
plt.figure(figsize=(16, 6))
plt.plot(simulated_future_dates, df_future['PREDICTED_DC_POWER'], label=f'Ապագա Կանխատեսում ({FUTURE_HOURS} ժամ)', color='mediumseagreen', linewidth=2)
plt.title('ՄԱՍ 2. Ապագայի Ավտոմատացված Կանխատեսում (Open-Meteo + XGBoost)', fontsize=14, fontweight='bold')
plt.xlabel('Ամսաթիվ և Ժամ', fontsize=12)
plt.ylabel('Հզորություն (DC Power)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.5, which='both')

ax2 = plt.gca()
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Part_2_Future_Forecast.png', dpi=300)
plt.show()

print("\nԱմբողջ գործընթացն ավարտված է:")