import pandas as pd

veri = {
    'Makine': ['A1', 'A2', 'A3', 'A1', 'A2'],
    'Ariza Süresi (dk)': [30, 45, 15, 35, 50],
    'Parça': ['Motor', 'Kayış', 'Piston', 'Motor', 'Kayış']
}

df = pd.DataFrame(veri)
print(df)


print("\n--- Ortalama Arıza Süresi (Makine Bazlı) ---")
print(df.groupby('Makine')['Ariza Süresi (dk)'].mean())

print("\n--- Arızaya Neden Olan Parça Sayısı ---")
print(df['Parça'].value_counts())


#Bu kütüphane grafik çizmek için kullanılır.
import matplotlib.pyplot as plt  

# Ortalama arıza süresini makine bazında çiz
#mean ortalama süreyi hesaplar
#plot sonuçları grafikte gösterir

# df.groupby('Makine')['Ariza Süresi (dk)'].mean().plot(
#     kind='bar',
#     title='Makine Bazlı Ortalama Arıza Süresi',
#     ylabel='Süre (dk)',
#     xlabel='Makine',
#     color='pink',
#     edgecolor='yellow'
# )

# Pasta grafiği çiz
df.groupby('Makine')['Ariza Süresi (dk)'].mean().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90,
    ylabel='',
    title='Makine Bazlı Arıza Süresi Oranı'
)

plt.tight_layout() # Grafik bileşenlerinin taşmasını engeller, yazılar ve başlıklar sıkışmaz.

plt.show()

