import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Տվյալների բեռնում և բաժանում (միայն մեր Առաջադեմ մոդելի համար)
print("Բեռնում ենք տվյալները...")
df = pd.read_csv('Final_Solar_Dataset.csv')

# Մեր հարստացված մուտքային տվյալները և թիրախը
X = df[['IRRADIATION', 'MODULE_TEMPERATURE', 'T2M', 'WS10M', 'RH2M', 'PRECTOTCORR']]
y = df['DC_POWER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Ստեղծում ենք բազային XGBoost մոդելը (առանց պարամետրերի)
xgb_model = XGBRegressor(random_state=42)

# 3. Սահմանում ենք պարամետրերի ցանցը (The Grid)
# Ծրագիրը կփորձարկի այս բոլոր թվերի բոլոր հնարավոր կոմբինացիաները
param_grid = {
    'n_estimators': [100, 200, 300],        # Ծառերի քանակը
    'max_depth': [3, 5, 7],                 # Յուրաքանչյուր ծառի առավելագույն խորությունը
    'learning_rate': [0.01, 0.05, 0.1],     # Ուսուցման արագությունը (քայլի չափը)
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