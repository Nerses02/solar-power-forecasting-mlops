import pandas as pd

# 1. ԿԱՐԴՈՒՄ ԵՆՔ ՖԱՅԼԵՐԸ
print("Կարդում ենք ֆայլերը...")
# Անվանումները փոխարինիր քո համակարգչում պահպանված ֆայլերի անուններով
df_gen = pd.read_csv('Plant_1_Generation_Data.csv')
df_wea = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
# Եթե աղյուսակի վերնագրերը 17-րդ տողում են, գրում ենք skiprows=16
df_nasa = pd.read_csv('POWER_Point_Hourly_20200515_20200618_019d99N_073d79E_LST.csv', skiprows=16)

# 2. ԱՄՍԱԹՎԵՐԻ ՍՏԱՆԴԱՐՏԱՑՈՒՄ (Data Cleaning)
print("Ստանդարտացնում ենք ամսաթվերի ֆորմատները...")

# Գեներացիայի ֆայլում ֆորմատը Օր-Ամիս-Տարի է
df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'], format='%d-%m-%Y %H:%M')

# Սենսորի ֆայլում ֆորմատը Տարի-Ամիս-Օր է
df_wea['DATE_TIME'] = pd.to_datetime(df_wea['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

# NASA-ի ֆայլում միավորում ենք 4 սյունակները մեկ DateTime սյունակի մեջ
df_nasa['DATE_TIME_NASA'] = pd.to_datetime(df_nasa[['YEAR', 'MO', 'DY', 'HR']].rename(
    columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'}))


# 3. ԱՂՅՈՒՍԱԿՆԵՐԻ ՄԻԱՑՈՒՄ (MERGE)
print("Միացնում ենք աղյուսակները...")

# ՔԱՅԼ 3.1: Միացնում ենք Kaggle-ի երկու ֆայլերը (Գեներացիա + Սենսոր)
# Միացնում ենք ըստ DATE_TIME-ի և PLANT_ID-ի
df_kaggle = pd.merge(df_gen, df_wea, on=['DATE_TIME', 'PLANT_ID'], suffixes=('_gen', '_wea'))

# ՔԱՅԼ 3.2: Պատրաստվում ենք միացնել NASA-ի տվյալները
# Քանի որ Kaggle-ի տվյալները կարող են լինել 15 րոպեն մեկ (օրինակ՝ 14:15, 14:30), 
# իսկ NASA-ն ունի միայն կլոր ժամեր (14:00), մենք ստեղծում ենք 'ժամանակի կլորացված' սյունակ
df_kaggle['HOUR_ROUNDED'] = df_kaggle['DATE_TIME'].dt.floor('h')

# ՔԱՅԼ 3.3: Միացնում ենք Kaggle-ի ամբողջական բազան NASA-ի բազային
df_final = pd.merge(df_kaggle, df_nasa, left_on='HOUR_ROUNDED', right_on='DATE_TIME_NASA', how='inner')


# 4. ՄԱՔՐՈՒՄ ԵՎ ՊԱՀՊԱՆՈՒՄ
print("Մաքրում ենք ավելորդ սյունակները և պահպանում վերջնական ֆայլը...")

# Ջնջում ենք ժամանակավոր կամ կրկնվող սյունակները, որոնք մեզ էլ պետք չեն
columns_to_drop = ['HOUR_ROUNDED', 'DATE_TIME_NASA', 'YEAR', 'MO', 'DY', 'HR']
df_final = df_final.drop(columns=columns_to_drop)

# Տեսնենք արդյունքը
print("\n--- Միացումը հաջողությամբ ավարտվեց ---")
print("Վերջնական աղյուսակի չափերը (տողեր, սյունակներ):", df_final.shape)
print("\nԱռաջին 3 տողերը հաստատման համար:")
print(df_final.head(3))

# Պահպանում ենք վերջնական սուպեր-բազան նոր CSV ֆայլում
df_final.to_csv('Final_Solar_Dataset.csv', index=False)
print("\nՏվյալները պահպանվել են 'Final_Solar_Dataset.csv' ֆայլում:")