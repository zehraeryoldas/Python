import pandas as pd
import joblib

# Modeli yükle
model = joblib.load('ariza_modeli.pkl')

# Kullanıcıdan veri al
makine = input("Makineyi gir (A1, A2, A3): ").upper()
parca = input("Parçayı gir (Motor, Kayış, Piston): ").capitalize()

# Boş form
girdi = {
    'Makine_A1': 0,
    'Makine_A2': 0,
    'Makine_A3': 0,
    'Parça_Kayış': 0,
    'Parça_Motor': 0,
    'Parça_Piston': 0
}

# Girdiyi doldur
girdi[f"Makine_{makine}"] = 1
girdi[f"Parça_{parca}"] = 1

# DataFrame oluştur
veri = pd.DataFrame([girdi])

# Tahmin yap
tahmin = model.predict(veri)
print(f"\nAI Tahmini: Bu arıza {tahmin[0]:.2f} dakika sürebilir.")
