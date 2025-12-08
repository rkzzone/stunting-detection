# 👶 Website Edukasi dan Deteksi Stunting

Sistem deteksi dini stunting berbasis web menggunakan **Streamlit**, **WHO Z-Score**, dan **Machine Learning (KNN)**.

---

## 📋 Daftar Isi

1. [Fitur Utama](#fitur-utama)
2. [Teknologi](#teknologi)
3. [Struktur Project](#struktur-project)
4. [Instalasi](#instalasi)
5. [Cara Menggunakan](#cara-menggunakan)
6. [Penjelasan Metode](#penjelasan-metode)
7. [FAQ](#faq)
8. [Troubleshooting](#troubleshooting)

---

## ✨ Fitur Utama

### 1. Dashboard Edukasi Stunting
- 📚 Informasi lengkap tentang stunting
- 🔍 Ciri-ciri dan gejala stunting
- ⚠️ Dampak jangka pendek dan panjang
- 🛡️ Cara pencegahan
- ⭐ **1000 Hari Pertama Kehidupan (HPK)** - Periode emas pencegahan
- 🏥 Panduan pengobatan dan penanganan

### 2. Deteksi Dini Stunting
- 📝 Form input data anak (umur, tinggi, jenis kelamin)
- 📈 **Analisis WHO Z-Score** - Standar internasional
- 🤖 **Analisis Model KNN** - Machine Learning prediction
- 💡 Rekomendasi tindakan berdasarkan hasil
- 📊 Visualisasi hasil dengan gauge chart dan bar chart

### 3. Analisis Komparatif
- ✅ Perbandingan hasil WHO vs Model KNN
- 📊 Persentase risiko stunting
- 🎯 Interpretasi yang mudah dipahami

---

## 🛠️ Teknologi

- **Python 3.8+**
- **Streamlit** - Web framework
- **Scikit-learn** - Machine Learning (KNN)
- **Pandas & NumPy** - Data processing
- **Plotly** - Interactive visualization
- **SciPy** - Scientific computing

---

## 📁 Struktur Project

```
C:\Users\muham\Project\Stunting\
│
├── data_balita.csv              # Dataset training
├── stunting_detection.ipynb     # Jupyter Notebook workflow lengkap
│
├── app.py                        # Main Streamlit application
├── z_score_calculator.py         # WHO Z-Score calculator
├── knn_model_trainer.py          # KNN model trainer
│
├── models/                       # Folder untuk saved models
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── model_metadata.pkl
│
├── requirements.txt              # Python dependencies
└── README.md                     # Dokumentasi ini
```

---

## 🚀 Instalasi

### Langkah 1: Persiapan Environment

```bash
# Buat virtual environment (opsional tapi disarankan)
python -m venv venv

# Aktivasi virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Langkah 2: Install Dependencies

```bash
# Install semua library yang diperlukan
pip install streamlit pandas numpy scikit-learn scipy openpyxl plotly

# Atau menggunakan requirements.txt
pip install -r requirements.txt
```

### Langkah 3: Verifikasi Instalasi

```bash
# Check Python version (harus 3.8+)
python --version

# Check Streamlit version
streamlit --version
```

---

## 📖 Cara Menggunakan

### Metode 1: Menggunakan Jupyter Notebook (Recommended)

1. **Buka VS Code** dan install extension **Jupyter**

2. **Buka file** `stunting_detection.ipynb`

3. **Jalankan cell per cell** dari atas ke bawah:
   - Cell 1-2: Import libraries
   - Cell 3-5: Load dan eksplorasi data
   - Cell 6-11: Training model KNN
   - Cell 12-13: Testing dan summary

4. **Setelah training selesai**, jalankan Streamlit app di terminal:
   ```bash
   cd C:\Users\muham\Project\Stunting
   streamlit run app.py
   ```

5. **Buka browser** di `http://localhost:8501`

### Metode 2: Training Manual (Tanpa Notebook)

1. **Training Model KNN**:
   ```bash
   cd C:\Users\muham\Project\Stunting
   python knn_model_trainer.py
   ```

2. **Jalankan Streamlit App**:
   ```bash
   streamlit run app.py
   ```

3. **Akses di browser**: `http://localhost:8501`

---

## 📚 Penjelasan Metode

### 1. WHO Z-Score Method

**Apa itu Z-Score?**
- Z-Score adalah nilai statistik yang menunjukkan seberapa jauh tinggi badan anak dari nilai median populasi referensi WHO
- Dihitung menggunakan metode **LMS (Lambda-Mu-Sigma)**

**Formula:**
```
Z = ((height/M)^L - 1) / (L * S)

Dimana:
- height = Tinggi badan anak (cm)
- L = Box-Cox transformation parameter
- M = Median reference value
- S = Coefficient of variation
```

**Kategori Status Gizi (WHO):**
- **Severely Stunted**: Z-score < -3 SD
- **Stunted**: -3 SD ≤ Z-score < -2 SD
- **Normal**: -2 SD ≤ Z-score ≤ +3 SD
- **Tall**: Z-score > +3 SD

**Keunggulan:**
✅ Standar internasional yang diakui WHO  
✅ Berdasarkan populasi referensi global  
✅ Digunakan oleh tenaga medis di seluruh dunia

### 2. K-Nearest Neighbors (KNN) Model

**Cara Kerja:**
1. Model membandingkan data anak dengan ribuan data training
2. Mencari K tetangga terdekat (most similar cases)
3. Menghitung probabilitas status gizi berdasarkan mayoritas tetangga
4. Memberikan persentase risiko stunting

**Hyperparameters (Optimized):**
- `n_neighbors`: Jumlah tetangga yang dipertimbangkan (3-15)
- `weights`: 'distance' atau 'uniform'
- `metric`: 'euclidean', 'manhattan', atau 'minkowski'

**Output:**
- **Prediksi Status**: severely stunted, stunted, normal, tall
- **Risk Percentage**: 0-100% (probabilitas stunting)
- **Confidence Level**: Tingkat kepercayaan model

**Keunggulan:**
✅ Belajar dari data populasi lokal  
✅ Memberikan probabilitas risiko  
✅ Dapat menangkap pola kompleks dalam data

---

## 🎯 Interpretasi Hasil

### Contoh Output

```
Input:
- Umur: 24 bulan
- Jenis Kelamin: Laki-laki
- Tinggi Badan: 80 cm

Output:

┌─────────────────────────────────┐
│   WHO Z-Score Analysis          │
├─────────────────────────────────┤
│ Z-Score: -2.5                   │
│ Status: Stunted (Pendek)        │
│ Interpretasi: TB di bawah       │
│ standar untuk usianya           │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│   KNN Model Analysis            │
├─────────────────────────────────┤
│ Prediksi: stunted               │
│ Risk: 65% (TINGGI)              │
│ Confidence: 78%                 │
└─────────────────────────────────┘

💡 REKOMENDASI:
1. Konsultasi ke Puskesmas/Bidan
2. Tingkatkan protein hewani
3. Berikan MPASI bergizi
4. Pantau rutin di Posyandu
```

### Decision Matrix

| Z-Score | KNN Risk | Action |
|---------|----------|--------|
| < -3 | > 70% | ⚠️ **URGENT** - Rujuk ke Spesialis Anak |
| -3 to -2 | 50-70% | ⚠️ **PERLU INTERVENSI** - Konsultasi Puskesmas |
| -2 to +3 | < 50% | ✅ **MONITOR** - Pertahankan gizi baik |
| > +3 | < 30% | ✅ **BAIK** - Terus pantau berkala |

---

## ❓ FAQ

### Q1: Apakah hasil deteksi ini 100% akurat?
**A:** Tidak. Ini adalah **screening tool awal** dengan akurasi ~85-90%. Diagnosis pasti harus dilakukan oleh tenaga medis profesional dengan pemeriksaan komprehensif.

### Q2: Kapan sebaiknya melakukan deteksi?
**A:** 
- Bayi/Balita: Setiap bulan di Posyandu
- Jika ada kekhawatiran: Segera
- Periode kritis: 0-24 bulan (1000 HPK)

### Q3: Apa yang harus dilakukan jika hasil menunjukkan stunting?
**A:**
1. **Jangan panik** - stunting bisa dicegah dan ditangani
2. **Konsultasi segera** ke Puskesmas/Bidan/Dokter
3. **Perbaiki gizi** - fokus pada protein hewani
4. **Follow-up rutin** - pantau perkembangan setiap bulan

### Q4: Mengapa ada dua metode analisis?
**A:** 
- **WHO Z-Score**: Standar medis internasional
- **KNN Model**: Memberikan perspektif tambahan dari data lokal
- Kombinasi keduanya memberikan gambaran lebih komprehensif

### Q5: Bisakah stunting disembuhkan?
**A:** Stunting adalah kondisi **ireversibel** (tidak bisa dikembalikan 100%), tetapi:
- ✅ Bisa **dicegah** sejak masa kehamilan
- ✅ Dampak bisa **diminimalkan** dengan intervensi dini
- ✅ Pertumbuhan bisa **dioptimalkan** dengan gizi baik

---

## 🔧 Troubleshooting

### Error: "Module not found"
```bash
# Install ulang dependencies
pip install --upgrade -r requirements.txt
```

### Error: "Model file not found"
```bash
# Training model terlebih dahulu
python knn_model_trainer.py
```

### Streamlit tidak bisa diakses
```bash
# Check port yang digunakan
streamlit run app.py --server.port 8502

# Atau clear cache
streamlit cache clear
```

### Dataset tidak ditemukan
```bash
# Pastikan file ada di path yang benar
# Cek: C:\Users\muham\Project\Stunting\data_balita.csv
```

### Model accuracy rendah
```bash
# Solusi:
# 1. Tambah data training
# 2. Feature engineering
# 3. Coba algoritma lain (Random Forest, SVM)
```

---

## 📊 Dataset Information

### Format Dataset (`data_balita.csv`)

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| Umur (bulan) | Integer | Usia anak dalam bulan | 0-60 |
| Jenis Kelamin | String | Laki-laki/Perempuan | - |
| Tinggi Badan (cm) | Float | Tinggi badan dalam cm | 40-150 |
| Status Gizi | String | severely stunted, stunted, normal, tall | - |

### Contoh Data:
```csv
Umur (bulan),Jenis Kelamin,Tinggi Badan (cm),Status Gizi
12,Laki-laki,75.5,normal
24,Perempuan,80.2,stunted
36,Laki-laki,95.8,normal
```

---

## 🎨 Color Palette

Website menggunakan color palette yang ramah dan menenangkan:

```
Primary:    #8FC0A9 (Teal)
Secondary:  #68B0AB (Darker Teal)
Accent:     #FAF3DD (Cream)
Success:    #10B981 (Green)
Warning:    #F59E0B (Orange)
Danger:     #DC2626 (Red)
```

---

## 📝 Development Notes

### Untuk Developer

**Improve Model:**
```python
# Tambahkan features
- Berat badan
- Riwayat ASI eksklusif
- Status ekonomi keluarga
- Riwayat penyakit

# Coba algoritma lain
- Random Forest
- Gradient Boosting
- Neural Networks
```

**Tambah Fitur:**
```python
# Export hasil ke PDF
# History tracking
# Multi-user support
# Admin dashboard
```

---

## 📞 Support & Contact

Untuk pertanyaan, bug report, atau saran:

- **Email**: [email@example.com]
- **GitHub**: [repository-url]
- **Puskesmas**: [nomor-telpon]

---

## 📜 License

Proyek ini dibuat untuk keperluan edukasi dan kesehatan masyarakat.

**Disclaimer:** Website ini adalah alat screening awal dan bukan pengganti diagnosa medis profesional.

---

## 🙏 Acknowledgments

- **WHO Child Growth Standards** - Data referensi Z-Score
- **Kementerian Kesehatan RI** - Panduan pencegahan stunting
- **Scikit-learn** - Machine Learning library
- **Streamlit** - Web framework

---

**⭐ Jika project ini bermanfaat, berikan star di GitHub!**

---

*Last updated: December 2024*