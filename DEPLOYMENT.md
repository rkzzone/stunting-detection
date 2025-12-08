# 🚀 Panduan Deployment Aplikasi Stunting Detection

## Pilihan 1: Streamlit Community Cloud (RECOMMENDED - GRATIS)

### Persiapan:

1. **Pastikan semua file sudah siap:**
   - ✅ `app.py` - File utama aplikasi
   - ✅ `requirements.txt` - Dependencies
   - ✅ `z_score_calculator.py` - Calculator module
   - ✅ `knn_model_trainer.py` - ML model trainer
   - ✅ `data_balita.csv` - Training data
   - ✅ `models/` folder dengan `knn_stunting_model.pkl` dan `label_encoder.pkl`
   - ✅ `.gitignore` - File yang tidak perlu di-push

### Langkah-langkah:

#### Step 1: Buat Repository di GitHub

1. Buka [GitHub](https://github.com) dan login
2. Klik tombol **"New"** atau **"+"** → **"New repository"**
3. Isi detail repository:
   - **Repository name**: `stunting-detection` (atau nama lain)
   - **Description**: `Web-based Stunting Detection System using WHO Z-Score and KNN`
   - **Public** atau **Private** (pilih Public untuk Streamlit Community Cloud gratis)
   - **Jangan** centang "Add a README file" (karena sudah ada)
4. Klik **"Create repository"**

#### Step 2: Push Code ke GitHub

Jalankan perintah berikut di terminal (dari folder project):

```powershell
# Inisialisasi git (jika belum)
git init

# Tambahkan semua file
git add .

# Commit
git commit -m "Initial commit: Stunting Detection System"

# Tambahkan remote repository (ganti USERNAME dan REPO_NAME)
git remote add origin https://github.com/USERNAME/REPO_NAME.git

# Push ke GitHub
git branch -M main
git push -u origin main
```

**Catatan**: Ganti `USERNAME` dengan username GitHub Anda dan `REPO_NAME` dengan nama repository yang dibuat di Step 1.

#### Step 3: Deploy ke Streamlit Community Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Klik **"Sign in with GitHub"**
3. Authorize Streamlit untuk mengakses GitHub Anda
4. Klik **"New app"**
5. Isi form deployment:
   - **Repository**: Pilih repository yang baru dibuat
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: (pilih nama unik, contoh: `stunting-detection-yourname`)
6. Klik **"Deploy!"**
7. Tunggu 2-5 menit sampai deployment selesai

#### Step 4: Verifikasi Deployment

- Aplikasi akan otomatis terbuka di browser
- Test semua fitur:
  - ✅ Dashboard loading dengan benar
  - ✅ Form deteksi bisa diisi
  - ✅ Analisis berjalan tanpa error
  - ✅ Chart dan visualisasi muncul

### Troubleshooting Streamlit Cloud:

**Jika ada error saat deploy:**

1. **Error: "No module named..."**
   - Pastikan package ada di `requirements.txt`
   - Cek typo pada nama package

2. **Error: "File not found"**
   - Pastikan semua file sudah di-push ke GitHub
   - Cek path file (gunakan relative path, bukan absolute)

3. **Error: Model tidak load**
   - Pastikan folder `models/` dan file `.pkl` sudah di-push
   - Cek di GitHub apakah file ada

4. **App terlalu lambat**
   - Gunakan `@st.cache_data` untuk cache data
   - Gunakan `@st.cache_resource` untuk load model

---

## Pilihan 2: Railway (Alternatif - Free Tier Available)

### Step 1: Buat akun Railway

1. Buka [railway.app](https://railway.app)
2. Sign up dengan GitHub

### Step 2: Deploy

1. Klik **"New Project"**
2. Pilih **"Deploy from GitHub repo"**
3. Pilih repository Anda
4. Railway akan auto-detect Python dan install dependencies

### Step 3: Konfigurasi

1. Tambahkan environment variable (jika ada)
2. Set start command:
   ```
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

---

## Pilihan 3: Render (Alternatif - Free Tier Available)

### Step 1: Buat akun Render

1. Buka [render.com](https://render.com)
2. Sign up dengan GitHub

### Step 2: Deploy

1. Klik **"New +"** → **"Web Service"**
2. Connect GitHub repository
3. Isi konfigurasi:
   - **Name**: `stunting-detection`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
4. Pilih **Free plan**
5. Klik **"Create Web Service"**

---

## Update Aplikasi yang Sudah Deploy

### Untuk Streamlit Community Cloud:

Setiap kali push ke GitHub, aplikasi akan otomatis update:

```powershell
git add .
git commit -m "Update: deskripsi perubahan"
git push
```

Atau manual reboot dari dashboard Streamlit Cloud:
1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Pilih app Anda
3. Klik **"Reboot app"** atau **"Manage app"** → **"Reboot"**

---

## Tips Optimasi Deployment:

1. **Gunakan caching:**
   ```python
   @st.cache_data
   def load_data():
       return pd.read_csv('data_balita.csv')
   
   @st.cache_resource
   def load_model():
       return joblib.load('models/knn_stunting_model.pkl')
   ```

2. **Compress model file** (jika terlalu besar):
   ```python
   # Saat save model
   joblib.dump(model, 'model.pkl', compress=3)
   ```

3. **Optimize dataset**: Jangan load data yang tidak perlu

4. **Monitor logs**: Check logs di platform deployment untuk debug

---

## Checklist Sebelum Deploy:

- [ ] Semua path menggunakan relative path (bukan `C:\Users\...`)
- [ ] `requirements.txt` berisi semua dependencies
- [ ] Model file (`.pkl`) ada dan ukurannya reasonable (<100MB)
- [ ] Data file (`.csv`) ada
- [ ] Test aplikasi di local berjalan dengan baik
- [ ] `.gitignore` sudah exclude file yang tidak perlu
- [ ] README.md informatif
- [ ] Tidak ada hardcoded secrets atau API keys
- [ ] Code sudah clean dan commented

---

## Custom Domain (Opsional):

Setelah deploy, Anda bisa setup custom domain:

**Streamlit Cloud:**
1. Buka App settings
2. Pilih "Custom domain"
3. Ikuti instruksi untuk setup CNAME di provider domain Anda

---

## Monitoring dan Maintenance:

1. **Check logs** secara berkala
2. **Update dependencies** untuk security patches
3. **Monitor usage** (terutama untuk free tier)
4. **Backup data** dan model secara berkala

---

**Butuh bantuan?** Hubungi support platform masing-masing atau cek dokumentasi resmi.
