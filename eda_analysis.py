import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Կարդում ենք մեր արդեն միացված վերջնական բազան
print("Բեռնում ենք տվյալները վերլուծության համար...")
df = pd.read_csv('Final_Solar_Dataset.csv')

# 2. Ստուգում ենք բազային ինֆորմացիան (կա՞ն արդյոք դատարկ վանդակներ)
print("\nԲացակայող (դատարկ) արժեքների քանակը ըստ սյունակների:")
print(df.isnull().sum())

# --- ԳՐԱՖԻԿ 1. Վահանակի Ջերմաստիճանի և Հզորության Կապը ---
# Տեսականորեն, շատ բարձր ջերմաստիճանը նվազեցնում է վահանակի ՕԳԳ-ն
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='MODULE_TEMPERATURE', y='DC_POWER', alpha=0.5, color='darkorange')
plt.title('Վահանակի Ջերմաստիճանի ազդեցությունը Արտադրված Հզորության վրա')
plt.xlabel('Վահանակի Ջերմաստիճան (°C)')
plt.ylabel('Արտադրված Հոսանք (DC Power)')
plt.grid(True)
plt.savefig('temperature_vs_power.png', dpi=300) # Պահպանում է նկարը թեզի համար
plt.show()

# --- ԳՐԱՖԻԿ 2. Կոռելյացիոն Մատրիցա ---
# Ցույց է տալիս, թե որ պարամետրն է ամենաշատը կապված գեներացիայի հետ
plt.figure(figsize=(12, 8))
numeric_cols = df[['DC_POWER', 'MODULE_TEMPERATURE', 
                   'IRRADIATION', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
corr_matrix = numeric_cols.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Մուտքային պարամետրերի և Գեներացիայի Կոռելյացիան (Correlation Matrix)')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300) # Պահպանում է նկարը թեզի համար
plt.show()