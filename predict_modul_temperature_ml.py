import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Բեռնում ենք տվյալները
print("Բեռնում ենք տվյալները...")
df = pd.read_csv('Final_Solar_Dataset.csv')

# 2. Նախապատրաստում ենք տվյալները ՋԵՐՄԱՍՏԻՃԱՆԻ մոդելի համար
# Այստեղ թիրախը ոչ թե հոսանքն է, այլ MODULE_TEMPERATURE-ը
X_temp = df[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
y_temp = df['MODULE_TEMPERATURE']

# Բաժանում ենք ուսուցման և թեստավորման
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# 3. Ստեղծում և Ուսուցանում ենք XGBoost-ը ջերմաստիճանի համար
print("Ուսուցանում ենք XGBoost մոդելը վահանակի ջերմաստիճանի համար...")
temp_model = XGBRegressor(learning_rate=0.1, max_depth=6, n_estimators=250, random_state=42)
temp_model.fit(X_train, y_train)

# 4. Գնահատում ենք մոդելի ճշտությունը
temp_preds = temp_model.predict(X_test)
mae_ml = mean_absolute_error(y_test, temp_preds)
print(f"\n--- ՄՈԴԵԼԻ ԱՐԴՅՈՒՆՔՆԵՐԸ ---")
print(f"Հին ֆիզիկական բանաձևի սխալանքը. ~7.36 °C")
print(f"ՆՈՐ XGBoost մոդելի սխալանքը (MAE): {mae_ml:.2f} °C")

# --- 5. ԳՐԱՖԻԿԱԿԱՆ ՀԱՄԵՄԱՏՈՒԹՅՈՒՆ ---
print("\nԿառուցում ենք վիզուալ համեմատության գրաֆիկը...")

# Վերցնում ենք մեկ ինվերտորի օրինակ՝ գրաֆիկի համար
inverter_id = df['SOURCE_KEY_gen'].unique()[0]
df_single = df[df['SOURCE_KEY_gen'] == inverter_id].copy()
df_single['DATE_TIME'] = pd.to_datetime(df_single['DATE_TIME'])
df_single = df_single.sort_values('DATE_TIME')

# Վերցնում ենք մոտ 10 օրվա հատված
df_plot = df_single.iloc[200:440].copy()

# Կանխատեսում ենք գրաֆիկի համար
X_plot = df_plot[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
y_ml_predicted = temp_model.predict(X_plot)

time_axis = df_plot['DATE_TIME']

plt.figure(figsize=(14, 6))

# Իրական Սենսորի ջերմաստիճանը (Կարմիր)
plt.plot(time_axis, df_plot['MODULE_TEMPERATURE'], 
         label='Իրական Սենսոր (Actual Module Temp)', color='crimson', linewidth=2)

# Մեր Մեքենայական Ուսուցման կանխատեսած ջերմաստիճանը (Կանաչ կետագծեր)
plt.plot(time_axis, y_ml_predicted, 
         label='XGBoost Կանխատեսում (ML Predicted Temp)', color='mediumseagreen', linewidth=2, linestyle='--')

plt.title('Արևային Վահանակի Ջերմաստիճան. Իրականություն ընդդեմ XGBoost Կանխատեսման', fontsize=14)
plt.xlabel('Ամսաթիվ և Ժամ', fontsize=12)
plt.ylabel('Ջերմաստիճան (°C)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.5)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('temperature_prediction_ml.png', dpi=300)
plt.show()