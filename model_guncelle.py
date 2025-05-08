import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# ✅ Genişletilmiş veri seti
df = pd.DataFrame({
    'Makine': ['A1', 'A2', 'A3', 'A1', 'A2', 'A3', 'A1', 'A3', 'A2', 'A1'],
    'Parça':  ['Motor', 'Kayış', 'Piston', 'Motor', 'Kayış', 'Piston', 'Motor', 'Kayış', 'Motor', 'Piston'],
    'Ariza Süresi': [30, 45, 15, 35, 50, 20, 38, 48, 41, 25]
})

# 🔄 Veriyi AI’nın anlayacağı hale getir
df_encoded = pd.get_dummies(df[['Makine', 'Parça']])
X = df_encoded
y = df['Ariza Süresi']

# 🧠 Modeli eğit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 💾 Modeli kaydet
joblib.dump(model, 'ariza_modeli.pkl')
print("✅ Yeni model başarıyla kaydedildi.")
