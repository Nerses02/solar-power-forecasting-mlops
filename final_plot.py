import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Բեռնում ենք տվյալները
print("Բեռնում ենք տվյալները...")
df = pd.read_csv('Final_Solar_Dataset.csv')

# 2. Առանձնացնում ենք մուտքային պարամետրերը և թիրախը ամբողջ բազայի համար
X = df[['IRRADIATION', 'MODULE_TEMPERATURE', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
y = df['DC_POWER']

# Բաժանում ենք ուսուցման և թեստավորման
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Ստեղծում և Ուսուցանում ենք մոդելը ԼԱՎԱԳՈՒՅՆ պարամետրերով
print("Ուսուցանում ենք մոդելը օպտիմիզացված պարամետրերով (սա կտևի մի քանի վայրկյան)...")
best_model = XGBRegressor(
    learning_rate=0.1, 
    max_depth=7, 
    n_estimators=300, 
    random_state=42
)
best_model.fit(X_train, y_train)

# 4. Գնահատում ենք մոդելը (համոզվելու համար, որ թվերը նույնն են)
final_preds = best_model.predict(X_test)
final_preds = np.maximum(0, final_preds) # Կտրում ենք բացասական թվերը

mae = mean_absolute_error(y_test, final_preds)
rmse = np.sqrt(mean_squared_error(y_test, final_preds))
r2 = r2_score(y_test, final_preds)

print("\n--- ՄՈԴԵԼԻ ԳՆԱՀԱՏԱԿԱՆԸ ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# --- 5. ԳՐԱՖԻԿԻ ԿԱՌՈՒՑՈՒՄ (25 օրվա մեծ սահմաններով) ---
print("\nԿառուցում ենք վերջնական մեծ գրաֆիկը...")

# Վերցնում ենք միայն մեկ ինվերտոր, որպեսզի գրաֆիկը խառնաշփոթ չլինի
inverter_id = df['SOURCE_KEY_gen'].unique()[0]
df_single = df[df['SOURCE_KEY_gen'] == inverter_id].copy()

# Ստանդարտացնում և դասավորում ենք ըստ ժամանակի
df_single['DATE_TIME'] = pd.to_datetime(df_single['DATE_TIME'])
df_single = df_single.sort_values('DATE_TIME')

# Վերցնում ենք 600 ժամվա (մոտ 25 օր) կտրվածք՝ համաձայն քո ցանկության
df_plot = df_single.iloc[100:700].copy()

# Առանձնացնում ենք գրաֆիկի մուտքային պարամետրերը
X_plot = df_plot[['IRRADIATION', 'MODULE_TEMPERATURE', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
y_actual = df_plot['DC_POWER']
time_axis = df_plot['DATE_TIME']

# Կանխատեսում ենք հենց այս 600 ժամվա համար
y_predicted = best_model.predict(X_plot)
y_predicted = np.maximum(0, y_predicted) # Բացասական թվերը դարձնում ենք 0

# Նկարում ենք
plt.figure(figsize=(16, 6))

# Իրական կորը (linewidth-ը սարքում ենք 1.5, որպեսզի շատ տվյալների մեջ գեղեցիկ երևա)
plt.plot(time_axis, y_actual.values, label='Իրական Արտադրություն (Actual)', color='royalblue', linewidth=1.5)

# Կանխատեսված կորը (linewidth-ը սարքում ենք 1.2 և կետագծերով)
plt.plot(time_axis, y_predicted, label='Մեր Կանխատեսումը (XGBoost Predicted)', color='darkorange', linewidth=1.2, linestyle='--')

# Ձևավորում
plt.title(f'Արևային Էներգիայի Կանխատեսման և Իրական Արտադրության Համեմատություն (Ինվերտոր: {inverter_id})', fontsize=14)
plt.xlabel('Ամսաթիվ և Ժամ (Date & Time)', fontsize=12)
plt.ylabel('Հզորություն (DC Power)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.4)

# Գեղեցկացնում ենք X առանցքի ամսաթվերը, որպեսզի չխառնվեն իրար
plt.xticks(rotation=45)
plt.tight_layout()

# Պահպանում և ցուցադրում
plt.savefig('actual_vs_predicted_25days.png', dpi=300)
plt.show()

print("\nԳրաֆիկը պահպանվել է 'actual_vs_predicted_25days.png' անվանումով:")