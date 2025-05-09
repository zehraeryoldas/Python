import pandas as pd
import joblib

# 💾 Eğitilmiş modeli yükle
model = joblib.load('ariza_modeli.pkl')

# 📥 CSV'den veri oku
df = pd.read_csv('girdi.csv')  # Dosya adın bu olacak

# 🔄 get_dummies ile AI'ya uygun formata çevir
df_encoded = pd.get_dummies(df)

# ✅ Gerekli tüm sütunlar varsa model hazır
# Modelin eğitildiği tüm sütunları sağlamalısın
gereken_sutunlar = [
    'Makine_A1', 'Makine_A2', 'Makine_A3',
    'Parça_Kayış', 'Parça_Motor', 'Parça_Piston'
]
for s in gereken_sutunlar:
    if s not in df_encoded.columns:
        df_encoded[s] = 0  # Eksikse 0 olarak ekle

# Sıra sırasına diz
df_encoded = df_encoded[gereken_sutunlar]

# 🤖 AI ile tahmin yap
tahminler = model.predict(df_encoded)
df['Tahmini Süre'] = tahminler

# 📤 Yeni CSV dosyasına yaz
df.to_csv('tahminli_sonuclar.csv', index=False)
print("✅ Tahminler başarıyla 'tahminli_sonuclar.csv' dosyasına yazıldı.")
