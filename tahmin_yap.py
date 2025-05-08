import joblib
import pandas as pd

# Kaydedilen modeli yükle
model = joblib.load('ariza_modeli.pkl')

# Yeni veri: A2 makinesi, Kayış parçası
yeni_veri = pd.DataFrame({
    'Makine_A1': [0],
    'Makine_A2': [1],
    'Makine_A3': [0],
    'Parça_Kayış': [1],
    'Parça_Motor': [0],
    'Parça_Piston': [0]
})

tahmin = model.predict(yeni_veri)
print(f"Tahmin edilen arıza süresi: {tahmin[0]:.2f} dakika")
