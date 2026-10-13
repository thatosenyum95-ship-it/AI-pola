# Ai pola

**Ai pola** adalah proyek pembelajaran machine learning yang dirancang agar bisa dikembangkan dari HP Android menggunakan GitHub.

## Fitur awal

1. **Klasifikasi teks** — model Naive Bayes sederhana yang belajar mengelompokkan teks ke kategori.
2. **Prediksi angka** — regresi linear dari contoh data.
3. **Pengenalan pola** — K-Means sederhana untuk mengelompokkan titik berdasarkan kemiripan.

> Ini adalah model pembelajaran dasar, bukan model bahasa besar seperti GPT. Model belajar dari dataset contoh yang disediakan dan hasilnya perlu diuji pada data baru.

## Struktur

```
AI-pola/
├── datasets/
│   ├── teks.csv
│   ├── angka.csv
│   └── pola.csv
├── models/                 # hasil model dibuat saat program dijalankan
├── train_text.py
├── train_numbers.py
├── train_patterns.py
└── .github/workflows/tests.yml
```

## Cara menjalankan

Program hanya menggunakan Python standard library, jadi tidak perlu memasang paket eksternal.

Di lingkungan yang bisa menjalankan Python:

```bash
python train_text.py
python train_numbers.py
python train_patterns.py
```

Model hasil pelatihan disimpan sebagai file JSON di folder `models/`.

## Menjalankan pemeriksaan lewat GitHub

Workflow GitHub Actions menjalankan ketiga skrip setiap kali ada push atau pull request. Buka tab **Actions** di repositori untuk melihat log dan status tes.

Catatan: GitHub menyimpan kode dan bisa menjalankan workflow otomatis dengan batas sumber daya tertentu. Ini tidak berarti GitHub menyediakan GPU untuk melatih model besar. Proyek awal ini sengaja memakai model kecil yang dapat dijalankan dengan CPU.

## Dataset

CSV di folder `datasets/` adalah data contoh untuk belajar. Tambahkan data yang relevan dan representatif sebelum menggunakan model untuk kebutuhan nyata. Hindari memasukkan informasi pribadi atau rahasia ke repositori publik.

## Lisensi

Belum ditetapkan. Tentukan lisensi sebelum membagikan atau menggunakan ulang proyek ini.
