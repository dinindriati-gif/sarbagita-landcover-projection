# Proyeksi Tutupan Lahan Kawasan Sarbagita (2020–2023–2026)

Dokumentasi metode dan alur kerja analisis klasifikasi serta proyeksi tutupan lahan untuk kawasan Sarbagita (Denpasar, Badung, Gianyar, Tabanan), menggunakan citra satelit RGB tahun 2020 dan 2023 sebagai basis, dan model *machine learning* untuk memproyeksikan kondisi tahun 2026.

## 1\. Ringkasan Proyek

Proyek ini bertujuan untuk:

1. Membangun titik sampel pelatihan (*training sample*) yang representatif dari area yang stabil pada periode 2020–2023.
2. Mengklasifikasikan tutupan lahan dari citra raster RGB tahun 2020 dan 2023 ke dalam 5 kelas tematik.
3. Memproyeksikan tutupan lahan tahun 2026 menggunakan model *Random Forest* berbasis tren perubahan 2020→2023.
4. Menguji validitas model proyeksi menggunakan metrik akurasi standar (Overall Accuracy, Cohen's Kappa, Producer's \& User's Accuracy).

## 2\. Kelas Tutupan Lahan

|Kode|Kelas|
|-|-|
|1|Hutan / Vegetasi Lebat|
|2|Pertanian / Sawah|
|3|Lahan Terbuka / Semak Belukar|
|4|Lahan Terbangun / Permukiman|
|5|Badan Air / Perairan|

## 3\. Data \& Tools

**Data input:**

* `Sarbagita\_2020.tif` — citra RGB tahun 2020
* `Sarbagita\_2023.tif` — citra RGB tahun 2023
* `danau\_batur.shp` — shapefile batas Danau Batur (digunakan untuk mengunci kelas Badan Air secara akurat di kawasan tersebut)

**Library Python:**

* `geopandas`, `rasterio`, `rioxarray` — pengolahan data geospasial (vektor \& raster)
* `numpy`, `pandas` — komputasi numerik dan tabulasi
* `scipy` (ndimage) — filter spasial
* `scikit-learn` — pemodelan *machine learning* dan evaluasi akurasi
* `openpyxl` — ekspor hasil ke format Excel

Instalasi dilakukan melalui Google Colab dengan Google Drive sebagai penyimpanan kerja (`/content/drive/MyDrive/ATRBPN`).

## 4\. Metodologi

### 4.1 Mount Google Drive \& Instalasi Library

Notebook dijalankan di Google Colab. *Google Drive* di-*mount* sebagai penyimpanan kerja agar data raster/vektor dan hasil keluaran tersimpan persisten, kemudian direktori kerja diarahkan ke folder proyek (`/content/drive/MyDrive/ATRBPN`). Seluruh *library* geospasial dan *machine learning* yang dibutuhkan (`geopandas`, `rasterio`, `rioxarray`, `openpyxl`, `scikit-learn`, dsb.) diinstal dan diimpor pada tahap ini.

### 4.2 Pembuatan Training Sample (Stratified Sampling Area Stabil 2020–2023)

Sebelum peta klasifikasi final dibuat, citra 2020 dan 2023 terlebih dahulu diklasifikasikan secara sementara menggunakan pendekatan **rule-based RGB thresholding** (kombinasi kecerahan dan rasio antar-kanal R/G/B), dengan pembatas spasial sederhana untuk badan air di 35% wilayah utara citra (lokasi Danau Batur).

Dari hasil klasifikasi sementara ini, diambil piksel yang **konsisten kelasnya** antara 2020 dan 2023 (area stabil) sebagai dasar titik sampel pelatihan, menggunakan **stratified random sampling** — maksimum 100 titik per kelas — agar representasi tiap kelas tutupan lahan seimbang. Setiap titik menyimpan atribut `id`, `class` (kode), dan `class\_name`, diekspor sebagai shapefile titik (`training\_sample\_tutupan\_lahan.shp`).

### 4.3 Klasifikasi Tutupan Lahan 2020 \& 2023 (Rule-Based RGB + Integrasi Shapefile Danau)

Klasifikasi final tutupan lahan 2020 dan 2023 disusun dengan pendekatan **rule-based thresholding** yang sama, namun disempurnakan dari versi pada tahap pembuatan *training sample*:

* **Masking area valid**: piksel dengan nilai R, G, B > 10 dianggap valid (mengeliminasi latar belakang/*NoData* hitam).
* **Badan Air (kelas 5)**: digabungkan dari dua sumber —

  * *Rule* spektral untuk laut/perairan umum (kecerahan rendah, kanal biru dominan terhadap merah dan hijau).
  * Shapefile `danau\_batur.shp` yang di-*rasterize* untuk mengunci Danau Batur secara presisi, **menggantikan** pendekatan pembatas spasial sederhana (35% wilayah utara) yang dipakai pada tahap pembuatan *training sample*.
* **Hutan/Vegetasi Lebat (kelas 1)**: kanal hijau dominan signifikan terhadap merah dan biru, dengan kecerahan relatif rendah (vegetasi rapat/pekat).
* **Pertanian/Sawah (kelas 2)**: kanal hijau sedikit lebih dominan, dengan rentang kecerahan sedang–tinggi yang lebih luas (ambang dilonggarkan dibanding versi awal agar menangkap variasi spektral sawah/kebun secara lebih baik).
* **Lahan Terbangun (kelas 4)**: kecerahan tinggi dengan selisih kanal R dan G kecil (karakteristik atap/material buatan).
* **Lahan Terbuka/Semak (kelas 3)**: kelas residual dari piksel valid yang tidak memenuhi kriteria kelas lain.

### 4.4 Post-Processing: Median Filter 3×3

Hasil klasifikasi piksel-per-piksel cenderung menghasilkan efek "*salt-and-pepper*" (piksel *noise* tersebar acak). Untuk merapikan hasil, diterapkan **median filter 3×3** pada peta klasifikasi 2020 dan 2023, dengan pengecualian: kelas Badan Air dikunci ulang setelah filter agar batas perairan tidak tergeser oleh proses *smoothing*. Hasil akhir disimpan sebagai `Sarbagita\_2020\_tutupan\_lahan.tif` dan `Sarbagita\_2023\_tutupan\_lahan.tif`.

### 4.5 Pelatihan Model Random Forest (Transisi 2020 → 2023)

Proyeksi 2026 dibangun dengan pendekatan **transisi tutupan lahan** menggunakan model **Random Forest Classifier**:

* **Fitur (*driving factors*)**:

  * Kelas tutupan lahan pada periode sebelumnya (2020, sebagai basis pelatihan)
  * Posisi relatif piksel (koordinat baris/kolom ternormalisasi, sebagai proksi arah utara–selatan dan timur–barat)
  * Jarak piksel dari titik pusat citra (proksi kedekatan terhadap pusat kawasan)
* **Target/label**: kelas tutupan lahan 2023
* **Pelatihan**: model dilatih pada hubungan transisi 2020→2023 menggunakan sub-sampel acak (maks. 100.000 piksel) agar proses efisien secara komputasi, dengan `n\_estimators=50`, `max\_depth=15`.

### 4.6 Prediksi / Proyeksi Tutupan Lahan 2026 (Berbasis Kondisi 2023)

Model yang telah dilatih pada transisi 2020→2023 diterapkan pada kondisi 2023 (sebagai basis "tahun sebelumnya") untuk memproyeksikan kelas tahun 2026, diproses secara *chunk* (per 500 baris) agar hemat memori. Piksel yang merupakan Badan Air pada 2023 dikunci tetap sebagai Badan Air pada hasil proyeksi 2026 (*restricting factor*), karena badan air relatif stabil dan tidak mengikuti logika transisi lahan darat. Hasil disimpan sebagai `Sarbagita\_2026\_proyeksi\_tutupan\_lahan.tif`.

Asumsi utama pendekatan ini: **pola transisi tutupan lahan periode 2020–2023 akan berlanjut dengan karakteristik spasial yang serupa hingga 2026** (proyeksi tren, bukan simulasi berbasis kebijakan/skenario).

### 4.7 Uji Akurasi Model (Train-Test Split, Confusion Matrix, Kappa)

Validitas model transisi diuji secara terpisah menggunakan skema **train-test split (80:20, stratified)** dari sub-sampel piksel 2020–2023 (bukan dari hasil proyeksi 2026, karena tidak tersedia data aktual 2026 sebagai pembanding). Model dilatih ulang dengan parameter yang lebih besar (`n\_estimators=100`, `max\_depth=20`) khusus untuk evaluasi ini. Metrik yang dihitung dari confusion matrix:

* **Overall Accuracy** — proporsi total piksel yang diklasifikasikan benar.
* **Cohen's Kappa Index** — tingkat kesesuaian klasifikasi setelah dikoreksi terhadap kemungkinan kesesuaian acak.
* **Producer's Accuracy** (per kelas) — probabilitas suatu kelas aktual diklasifikasikan dengan benar (kebalikan dari *omission error*).
* **User's Accuracy** (per kelas) — probabilitas suatu piksel yang diklasifikasikan ke suatu kelas benar-benar termasuk kelas tersebut di lapangan (kebalikan dari *commission error*).

> \*\*Catatan\*\*: nilai numerik hasil akurasi (Overall Accuracy, Kappa, Producer's/User's Accuracy per kelas) bergantung pada eksekusi aktual skrip terhadap data raster — isi tabel hasil dapat dilihat langsung pada file CSV/Excel keluaran setelah proses dijalankan.

### 4.8 Ekspor Hasil

Seluruh keluaran akhir diekspor: peta klasifikasi/proyeksi dalam format GeoTIFF (2020, 2023, 2026) serta tabulasi uji akurasi dalam format CSV dan Excel (`tabulasi\_uji\_akurasi\_sarbagita.csv` / `.xlsx`).

## 5\. Alur Kerja (Workflow)

```
1. Mount Google Drive \& instalasi library
        ↓
2. Pembuatan training sample (stratified sampling area stabil 2020–2023)
        ↓
3. Klasifikasi 2020 \& 2023 (rule-based RGB + integrasi shapefile danau)
        ↓
4. Post-processing: median filter 3x3
        ↓
5. Pelatihan model Random Forest (transisi 2020 → 2023)
        ↓
6. Prediksi/proyeksi tutupan lahan 2026 (berbasis kondisi 2023)
        ↓
7. Uji akurasi model (train-test split, confusion matrix, Kappa)
        ↓
8. Ekspor hasil: GeoTIFF (2020, 2023, 2026) + CSV/Excel akurasi
```

## 6\. Output yang Dihasilkan

|File|Deskripsi|
|-|-|
|`training\_sample\_tutupan\_lahan.shp`|Titik sampel pelatihan stratified per kelas|
|`Sarbagita\_2020\_tutupan\_lahan.tif`|Peta klasifikasi tutupan lahan 2020|
|`Sarbagita\_2023\_tutupan\_lahan.tif`|Peta klasifikasi tutupan lahan 2023|
|`Sarbagita\_2026\_proyeksi\_tutupan\_lahan.tif`|Peta proyeksi tutupan lahan 2026|
|`tabulasi\_uji\_akurasi\_sarbagita.csv` / `.xlsx`|Tabulasi Overall Accuracy, Cohen's Kappa, Producer's \& User's Accuracy|

## 7\. Keterbatasan Metode

* Klasifikasi berbasis RGB murni (tanpa band inframerah/NIR) memiliki keterbatasan dalam membedakan kelas dengan karakteristik spektral tampak yang mirip (mis. lahan terbuka vs. lahan terbangun tertentu, atau vegetasi kering vs. lahan terbuka), sehingga ambang batas (*threshold*) disusun secara empiris dan mungkin perlu penyesuaian untuk kawasan/citra lain.
* Titik sampel pelatihan dibangun dari hasil klasifikasi sementara (sebelum penyempurnaan dengan shapefile danau), sehingga sebagian kecil titik pada area transisi batas badan air berpotensi tidak sepenuhnya konsisten dengan peta klasifikasi final.
* Proyeksi 2026 bersifat **ekstrapolasi tren spasial-statistik**, bukan model berbasis proses fisik/kebijakan (mis. tidak memperhitungkan rencana tata ruang, kebijakan pembangunan, atau faktor sosial-ekonomi secara eksplisit).
* Validasi model dilakukan terhadap data 2020–2023 (bukan data aktual 2026), sehingga metrik akurasi mencerminkan performa model pada tugas transisi historis, sebagai indikator kepercayaan terhadap proyeksi ke depan.

