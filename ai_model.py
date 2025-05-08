import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Veri seti
df = pd.DataFrame({
    'Makine': ['A1', 'A2', 'A3', 'A1', 'A2', 'A3', 'A1'],
    'Parça': ['Motor', 'Kayış', 'Piston', 'Motor', 'Kayış', 'Piston', 'Motor'],
    'Ariza Süresi': [30, 45, 15, 35, 50, 20, 38]
})

# Kategorik veriyi sayıya çevir
df_encoded = pd.get_dummies(df[['Makine', 'Parça']])
X = df_encoded
y = df['Ariza Süresi']

# Veriyi ayır (eğitim ve test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluştur
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yap
yeni_veri = pd.DataFrame({
    'Makine_A1': [1],
    'Makine_A2': [0],
    'Makine_A3': [0],
    'Parça_Kayış': [0],
    'Parça_Motor': [1],
    'Parça_Piston': [0]
})

tahmin = model.predict(yeni_veri)
print(f"AI Tahmini: Bu arıza {tahmin[0]:.2f} dakika sürecek.")


import joblib
joblib.dump(model, 'ariza_modeli.pkl')  # kaydet
model = joblib.load('ariza_modeli.pkl')  # tekrar kullan


