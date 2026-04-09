import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Տվյալների բեռնում
print("Բեռնում ենք տվյալները...")
df = pd.read_csv('Final_Solar_Dataset.csv')

# 2. Նպատակային փոփոխական (Target) - Այն, ինչ ուզում ենք գուշակել
y = df['DC_POWER']

# 3. Բազային (Baseline) մոդելի մուտքային տվյալներ (Միայն Արևի ճառագայթում)
# Սա նմանակում է ստանդարտ հասարակ ծրագրերը
X_baseline = df[['IRRADIATION']]

# 4. Մեր Առաջադեմ (Advanced) մոդելի մուտքային տվյալներ (Սենսորային միաձուլում)
# Ներառում է թե՛ միկրո, թե՛ մակրո կլիմայական գործոնները
X_advanced = df[['IRRADIATION', 'MODULE_TEMPERATURE',
                'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]

# 5. Տվյալների բաժանում Ուսուցման (80%) և Թեստավորման (20%)
# random_state=42-ը ապահովում է, որ երկու մոդելներն էլ քննություն հանձնեն նույն 20% անծանոթ տվյալների վրա
print("Բաժանում ենք տվյալները ուսուցման և թեստավորման համար...")
X_base_train, X_base_test, y_train, y_test = train_test_split(X_baseline, y, test_size=0.2, random_state=42)
X_adv_train, X_adv_test, _, _ = train_test_split(X_advanced, y, test_size=0.2, random_state=42)

# --- ՄՈԴԵԼ 1. ԲԱԶԱՅԻՆ (Գծային Ռեգրեսիա) ---
print("\nՈւսուցանում ենք Բազային մոդելը (Linear Regression)...")
base_model = LinearRegression()
base_model.fit(X_base_train, y_train)
base_preds = base_model.predict(X_base_test)

# --- ՄՈԴԵԼ 2. ՄԵՐ ԱՌԱՋԱՐԿԱԾԸ (XGBoost - Ոչ գծային ծառեր) ---
print("Ուսուցանում ենք Հիբրիդային մոդելը (XGBoost)...")
adv_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
adv_model.fit(X_adv_train, y_train)
adv_preds = adv_model.predict(X_adv_test)

# --- ԳՆԱՀԱՏՈՒՄ ԵՎ ՀԱՄԵՄԱՏՈՒԹՅՈՒՆ ---
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n[{model_name}]")
    print(f"MAE  (Միջին բացարձակ սխալ): {mae:.2f}")
    print(f"RMSE (Արմատ միջին քառակուսային սխալ): {rmse:.2f}")
    print(f"R²   (Ճշտության գործակից 0-ից 1): {r2:.4f}")
    return mae, rmse, r2

base_mae, base_rmse, base_r2 = evaluate_model(y_test, base_preds, "ԲԱԶԱՅԻՆ ՄՈԴԵԼ (Միայն Արև)")
adv_mae, adv_rmse, adv_r2 = evaluate_model(y_test, adv_preds, "ԱՌԱՋԱԴԵՄ ՄՈԴԵԼ (Մեր XGBoost-ը)")

# --- ԳՐԱՖԻԿԱԿԱՆ ԱՊԱՑՈՒՅՑ ԹԵԶԻ ՀԱՄԱՐ ---
labels = ['MAE (Սխալանք)', 'RMSE (Սխալանք)']
base_errors = [base_mae, base_rmse]
adv_errors = [adv_mae, adv_rmse]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, base_errors, width, label='Բազային Մոդել (Պարզ)', color='lightcoral')
rects2 = ax.bar(x + width/2, adv_errors, width, label='Մեր Մոդելը (XGBoost)', color='mediumseagreen')

ax.set_ylabel('Սխալանքի չափը (որքան ցածր, այնքան լավ)')
ax.set_title('Գոյություն ունեցող և մեր առաջարկած մոդելների համեմատություն')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Ավելացնում ենք թվերը սյուների վրա
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.0f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 միավոր դեպի վեր
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_comparison_bar_chart.png', dpi=300)
plt.show()