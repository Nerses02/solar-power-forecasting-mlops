import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Կարդում ենք վերջնական բազան
df = pd.read_csv('Final_Solar_Dataset.csv')

# 2. Նախապատրաստում ենք տվյալները կորի համար
# Որպեսզի կորը սահուն լինի, մեզ պետք են բոլոր հնարավոր ջերմաստիճանները
X = df['MODULE_TEMPERATURE']
Y = df['DC_POWER']

# 3. Պոլինոմիալ ֆիտտինգ (2-րդ աստիճանի կոր)`
# np.polyfit-ը հաշվում է կորի գործակիցները, որոնք լավագույնս են նկարագրում կետերը
p = np.polyfit(X, Y, 2)

# Ստեղծում ենք ֆունկցիա այդ գործակիցների հիման վրա
polynomial = np.poly1d(p)

# Ստեղծում ենք ջերմաստիճանների սահուն շարք` ամենացածրից մինչև ամենաբարձրը
x_line = np.linspace(X.min(), X.max(), 100)

# Հաշվում ենք հզորությունը այդ սահուն ջերմաստիճանների համար` օգտագործելով մեր պոլինոմը
y_line = polynomial(x_line)

# 4. Կառուցում ենք գրաֆիկը
plt.figure(figsize=(10, 6))

# Առաջինը նկարում ենք ռեալ կետերը (scatterplot)` alpha=0.3` որպեսզի կորը երևա
plt.scatter(X, Y, alpha=0.3, color='darkorange', label='Ռեալ Տվյալներ')

# Ավելացնում ենք պոլինոմիալ կորը` linewidth=3 և հաստ գույնով
plt.plot(x_line, y_line, color='blue', linewidth=3, label='Պոլինոմիալ Տրենդ')

# Ավելացնում ենք նկարագրությունները
plt.title('Վահանակի Ջերմաստիճանի և Հզորության Կապը` Efficiency Drop-ի Visualisation')
plt.xlabel('Վահանակի Ջերմաստիճան (°C)')
plt.ylabel('Արտադրված Հոսանք (DC Power)')
plt.legend()
plt.grid(True)

# 5. Պահպանում ենք նոր նկարը
plt.savefig('temperature_vs_power_with_curve.png', dpi=300)
plt.show()