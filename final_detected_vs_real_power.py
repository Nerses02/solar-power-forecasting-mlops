import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Տվյալների բեռնում և բաժանում (միայն մեր Առաջադեմ մոդելի համար)
print("Բեռնում ենք տվյալները...")
df = pd.read_csv('Final_Solar_Dataset.csv')

# Մեր հարստացված մուտքային տվյալները և թիրախը
X = df[['IRRADIATION', 'MODULE_TEMPERATURE', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
y = df['DC_POWER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Ստեղծում ենք բազային XGBoost մոդելը (առանց պարամետրերի)
xgb_model = XGBRegressor(random_state=42)

# 3. Սահմանում ենք պարամետրերի ընդլայնված ցանցը (The Extended Grid)
# Այս ցանցը կփորձարկի 192 տարբեր կոմբինացիաներ (ընդհանուր 576 վարժեցում cv=3-ի դեպքում)
param_grid = {
    'n_estimators': [300, 500, 800],           # Շարունակում ենք աճեցնել ծառերի քանակը
    'max_depth': [5, 7, 9, 12],                # Թույլ ենք տալիս ծառերին ավելի խորը մտածել
    'learning_rate': [0.01, 0.05, 0.1, 0.2],   # Ուսուցման արագության ավելի շատ տարբերակներ
    'subsample': [0.8, 1.0],                   # Ամեն ծառ կօգտագործի տվյալների 80% կամ 100%-ը (կանխում է overfitting-ը)
    'colsample_bytree': [0.8, 1.0]             # Ամեն ծառ կնայի սյունակների 80% կամ 100%-ին
}

# 4. Սկսում ենք Ցանցային Որոնումը (Grid Search)
# cv=3 նշանակում է Խաչաձև ստուգում (Cross-Validation) 3 մասով՝ արդյունքների հավաստիությունն ապահովելու համար
print("\nՍկսում ենք Գերպարամետրերի օպտիմիզացիան (սա կարող է տևել մի քանի րոպե)...")
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           cv=3, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)

# Ստիպում ենք համակարգչին սովորել և գտնել լավագույնը
grid_search.fit(X_train, y_train)

# 5. Արդյունքների արտածում
print("\n--- ՕՊՏԻՄԻԶԱՑԻԱՅԻ ԱՐԴՅՈՒՆՔՆԵՐԸ ---")
best_params = grid_search.best_params_
print(f"Լավագույն պարամետրերը մեր տվյալների համար:\n{best_params}")

# 6. Թեստավորում ենք այդ լավագույն (Օպտիմիզացված) մոդելը անծանոթ տվյալների վրա
best_model = grid_search.best_estimator_
final_preds = best_model.predict(X_test)

# Հաշվում ենք նոր, լավացված սխալանքները
mae = mean_absolute_error(y_test, final_preds)
rmse = np.sqrt(mean_squared_error(y_test, final_preds))
r2 = r2_score(y_test, final_preds)

print("\n--- ԼԱՎԱՑՎԱԾ ՄՈԴԵԼԻ ՎԵՐՋՆԱԿԱՆ ԳՆԱՀԱՏԱԿԱՆԸ ---")
print(f"MAE  (Միջին բացարձակ սխալ): {mae:.2f}")
print(f"RMSE (Արմատ միջին քառակուսային սխալ): {rmse:.2f}")
print(f"R²   (Ճշտության գործակից): {r2:.4f}")

# --- 7. ԻՐԱԿԱՆՈՒԹՅՈՒՆ ԸՆԴԴԵՄ ԿԱՆԽԱՏԵՍՄԱՆ (Time-Series Plot) ---
print("\nԿառուցում ենք վերջնական կանխատեսման գրաֆիկը...")

# Վերցնում ենք բազայի վերջին 100 տողերը (մոտ 4 օրվա անընդմեջ տվյալներ)` 
# որպեսզի գրաֆիկը լինի պարզ և հասկանալի
df_plot = df.tail(100).copy()

# Առանձնացնում ենք մուտքային պարամետրերը և իրական հոսանքը այս 100 ժամվա համար
X_plot = df_plot[['IRRADIATION', 'MODULE_TEMPERATURE', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
y_actual = df_plot['DC_POWER']

# Խնդրում ենք մեր ՕՊՏԻՄԻԶԱՑՎԱԾ մոդելին գուշակել այս 100 ժամվա հոսանքը
y_predicted = best_model.predict(X_plot)

# Կառուցում ենք գեղեցիկ գրաֆիկ
plt.figure(figsize=(14, 6))

# 1. Նկարում ենք Իրական կորը (Կապույտ և հաստ գծով)
plt.plot(y_actual.values, label='Իրական Արտադրություն (Actual)', color='royalblue', linewidth=2.5)

# 2. Նկարում ենք Կանխատեսված կորը (Նարնջագույն, կետագծերով, որպեսզի տեսանելի լինի նախորդի վրայից)
plt.plot(y_predicted, label='Մեր Կանխատեսումը (XGBoost Predicted)', color='darkorange', linewidth=2, linestyle='--')

# Գրաֆիկի ձևավորում
plt.title('Արևային Էներգիայի Կանխատեսման և Իրական Արտադրության Համեմատություն (100 ժամվա կտրվածքով)', fontsize=14)
plt.xlabel('Ժամանակ (Ժամեր)', fontsize=12)
plt.ylabel('Հզորություն (DC Power)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.5)

# Պահպանում և ցուցադրում
plt.tight_layout()
plt.savefig('actual_vs_predicted_timeseries.png', dpi=300)
plt.show()