import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Բեռնում ենք տվյալները վերլուծության համար...")
df = pd.read_csv('Final_Master_Dataset.csv')

# --- Կոռելյացիոն Մատրիցա ---
plt.figure(figsize=(14, 10))

# Ընտրում ենք միայն գեներացիայի և եղանակային թվային սյուները (առանց ID-ների և ամսաթվի)
numeric_cols = df[['DC_POWER', 'AC_POWER', 'MODULE_TEMPERATURE', 'IRRADIATION', 
                   'ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DNI', 'ALLSKY_SFC_SW_DIFF', 
                   'T2M', 'WS10M', 'RH2M']]

# Անվանումները դարձնում ենք պրոֆեսիոնալ և հասկանալի գրաֆիկի համար
rename_dict = {
    'DC_POWER': 'DC Power',
    'AC_POWER': 'AC Power',
    'MODULE_TEMPERATURE': 'Module Temp (Sensor)',
    'IRRADIATION': 'Irrad (Sensor)',
    'ALLSKY_SFC_SW_DWN': 'GHI (NASA)',
    'ALLSKY_SFC_SW_DNI': 'DNI (NASA)',
    'ALLSKY_SFC_SW_DIFF': 'DHI (NASA)',
    'T2M': 'Air Temp (NASA)',
    'WS10M': 'Wind Speed',
    'RH2M': 'Humidity'
}

numeric_cols = numeric_cols.rename(columns=rename_dict)
corr_matrix = numeric_cols.corr()

# Կառուցում ենք գրաֆիկը (vmin=-1 և vmax=1 ֆիքսված սանդղակով)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
plt.title('Մուտքային պարամետրերի և Գեներացիայի Կոռելյացիան', pad=20, fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Պահպանում ենք պատրաստի նկարը images թղթապանակում
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight') 
plt.show()