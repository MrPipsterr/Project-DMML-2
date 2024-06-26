# Project-DMML-2

## Anggota Kelompok : 
  1. H071221102 - Muh. Yusuf Fikry
  2. H071221070 - Zefanya Farel Palinggi
  3. H071221078 - Andi Adnan
  4. H071211062 - Muh. Taufiqurrahman



## Deskripsi Project 
Aplikasi Flood Prediction dibuat dengan tujuan penggunaan oleh sebuah organisasi atau  lembaga daerah guna memprediksi seberapa besar peluang banjir terjadi di daerahnya. Aplikasi ini akan menerima input nilai-nilai berdasarkan faktor penyebab banjir yang datanya mungkin hanya dimengerti oleh orang yang bekerja di lembaga terkait. Harapannya, dengan nilai prediksi tersebut, dapat membantu upaya mitigasi dan kesiapsiagaan akan bencana banjir.

##### Sumber Dataset
[Flood Prediction Factor on Kaggle](https://www.kaggle.com/competitions/playground-series-s4e5/data)

##### Aplikasi ini dibuat menggunakan : 
pyqt5

## Algoritma Machine Learning yang Digunakan :
  1. Support Vector Machine
  2. Bayesian Ridge Regression
  3. Catboost Regressor
  4. K-Nearest Neighbors Regressor
  5. Light GBM Regressor
  6. Linear Regression

## Fitur Aplikasi
![Screenshot_2024-06-25_133307](https://github.com/yusuffikry/Project-DMML-2/assets/113654172/1d493fe7-b8d0-4b60-a84a-edf6031fc959)
![image](https://github.com/yusuffikry/Project-DMML-2/assets/113654172/073c86a0-b748-4838-9313-8d84d8bb1387)
  - input nilai fitur faktor terjadinya banjir
  - pemilihan model machine learning yang akan digunakan untuk prediksi
  - kalkulasi peluang terjadinya banjir
  - halaman info untuk melihat keterangan mengenai performa model

## Cara Menjalankan Aplikasi
Untuk menjalankan Aplikasi Flood Prediction, perlu untuk melakukan clone pada repositori berikut

```bash
git clone https://github.com/yusuffikry/Project-DMML-2.git
```

Selanjutnya install beberapa library /modul dari bahasa pemrograman Python yang dibutuhkan

```bash
pip install -r requirements.txt
```

Mulai jalankan aplikasi dengan menjalankan command line berikut pada direktori utama project

```bash
python app.py
```
