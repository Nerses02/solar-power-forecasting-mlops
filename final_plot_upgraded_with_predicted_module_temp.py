import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ==========================================
# ՓՈՒԼ 1. ՏՎՅԱԼՆԵՐԻ ԲԵՌՆՈՒՄ ԵՎ ՈՒՍՈՒՑՈՒՄ
# ==========================================
print("Բեռնում ենք տվյալները...")
df = pd.read_csv('Final_Solar_Dataset.csv')

# --- ՄՈԴԵԼ 1: Ջերմաստիճան Գուշակող ---
print("Ուսուցանում ենք Մոդել 1-ը (Վիրտուալ Ջերմաչափ)...")
X_weather = df[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
y_temp = df['MODULE_TEMPERATURE']

# Ստեղծում և սովորեցնում ենք (օգտագործում ենք լավ պարամետրեր)
temp_model = XGBRegressor(learning_rate=0.1, max_depth=6, n_estimators=250, random_state=42)
temp_model.fit(X_weather, y_temp)

# --- ՄՈԴԵԼ 2: Հոսանք Գուշակող (Գլխավոր Մոդել) ---
print("Ուսուցանում ենք Մոդել 2-ը (Հզորության Գուշակող)...")
# Սա սովորում է պատմական իրական տվյալների վրա, որ հասկանա ճիշտ ֆիզիկան
X_full = df[['IRRADIATION', 'MODULE_TEMPERATURE', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
y_power = df['DC_POWER']

# Մեր գտած լավագույն պարամետրերով
power_model = XGBRegressor(learning_rate=0.1, max_depth=7, n_estimators=300, random_state=42)
power_model.fit(X_full, y_power)


# ==========================================
# ՓՈՒԼ 2. ԿԱՍԿԱԴԱՅԻՆ ԿԱՆԽԱՏԵՍՈՒՄ (Ռեալ Աշխատանքի Սիմուլյացիա)
# ==========================================
print("\nՍկսում ենք Երկփուլ Կանխատեսումը 25 օրվա կտրվածքով...")

# 1. Առանձնացնում ենք մեկ ինվերտորի 25 օրվա (600 ժամվա) հատված
inverter_id = df['SOURCE_KEY_gen'].unique()[0]
df_plot = df[df['SOURCE_KEY_gen'] == inverter_id].copy()
df_plot['DATE_TIME'] = pd.to_datetime(df_plot['DATE_TIME'])
df_plot = df_plot.sort_values('DATE_TIME').iloc[100:700].copy()

time_axis = df_plot['DATE_TIME']
y_actual_power = df_plot['DC_POWER']

# 2. Վերցնում ենք ՄԻԱՅՆ ԵՂԱՆԱԿԸ այս 600 ժամվա համար (կարծես սենսոր չունենք)
X_future_weather = df_plot[['IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]

# 3. ՔԱՅԼ Ա (Կասկադ): Գուշակում ենք Վահանակի ջերմաստիճանը
predicted_module_temp = temp_model.predict(X_future_weather)

# 4. ՔԱՅԼ Բ (Կասկադ): Հավաքում ենք նոր մուտքային բազան Մոդել 2-ի համար
# Ստեղծում ենք X_future աղյուսակը և ճիշտ տեղում (index 1) ավելացնում ենք ՄԵՐ ԳՈՒՇԱԿԱԾ ջերմաստիճանը
X_future = X_future_weather.copy()
X_future.insert(1, 'MODULE_TEMPERATURE', predicted_module_temp)

# 5. ՔԱՅԼ Գ (Վերջնական): Գուշակում ենք Հոսանքը՝ հիմնվելով վիրտուալ ջերմաստիճանի վրա
y_cascaded_power = power_model.predict(X_future)
y_cascaded_power = np.maximum(0, y_cascaded_power) # Բացասական թվերը դարձնում ենք 0


# ==========================================
# ՓՈՒԼ 3. ԳՐԱՖԻԿԱԿԱՆ ԱՊԱՑՈՒՅՑ
# ==========================================
print("Կառուցում ենք գրաֆիկը...")

plt.figure(figsize=(16, 6))

# Իրական հոսանքը
plt.plot(time_axis, y_actual_power.values, 
         label='Իրական Արտադրություն (Actual)', color='royalblue', linewidth=1.5)

# Մեր կասկադային (առանց սենսորի) կանխատեսումը
plt.plot(time_axis, y_cascaded_power, 
         label='Կասկադային Կանխատեսում (ML Temp + ML Power)', color='darkorange', linewidth=1.2, linestyle='--')

plt.title(f'Ամբողջապես Ավտոմատացված Կանխատեսում (Առանց Ֆիզիկական Սենսորի)', fontsize=14)
plt.xlabel('Ամսաթիվ և Ժամ (Date & Time)', fontsize=12)
plt.ylabel('Հզորություն (DC Power)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.4)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('fully_cascaded_prediction.png', dpi=300)
plt.show()

print("\nԳրաֆիկը պահպանվել է 'fully_cascaded_prediction.png' անվանումով:")