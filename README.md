# Ai Pola

Ai Pola adalah proyek edukasi untuk analisis urutan angka 4 digit dan contoh model machine learning sederhana. Model tidak menjamin hasil berikutnya, terutama untuk data acak atau independen.

## Fitur

- Validasi data empat digit dengan nol di depan tetap dipertahankan.
- Kandidat baseline Python dan website: digit dengan frekuensi tertinggi pada tiap posisi; seri diputuskan berdasarkan kemunculan terbaru di antara digit seri.
- Walk-forward backtest: setiap prediksi hanya menggunakan data yang tersedia sebelumnya.
- Ringkasan frekuensi per posisi digit.
- Penyimpanan lokal browser; data tidak dikirim ke server.
- Contoh edukasi terpisah: Naive Bayes untuk teks, regresi linear, dan K-Means.
- Unit test dan GitHub Actions.

## Batasan metodologi

Baseline frekuensi per posisi tidak mempelajari dependensi temporal dan bukan bukti bahwa hasil masa lalu memengaruhi hasil berikutnya. Untuk empat digit seragam dan independen, peluang kecocokan tepat suatu kombinasi adalah 1/10.000 (0,01%). Backtest yang pendek mudah menghasilkan akurasi yang menyesatkan. Bandingkan dengan baseline yang sesuai dan gunakan periode out-of-sample sebelum menarik kesimpulan.

Model teks, regresi, dan clustering adalah contoh edukasi terpisah; mereka tidak terintegrasi ke prediksi angka 4 digit. Dataset contoh yang kecil tidak cukup untuk menyimpulkan kualitas generalisasi.

## Menggunakan website

Aktifkan GitHub Pages di Settings → Pages, lalu buka [AI-pola GitHub Pages](https://thatosenyum95-ship-it.github.io/AI-pola/) jika tersedia. Website dapat dibuka dari `index.html` juga.

1. Masukkan riwayat dari data paling lama ke paling baru, satu angka empat digit per baris atau dipisahkan spasi/koma/titik koma.
2. Tekan **Baca Pola** untuk melihat kandidat baseline, frekuensi digit, dan evaluasi walk-forward.
3. Tambahkan satu hasil aktual di bagian **Tambah hasil terbaru**.
4. Data disimpan pada local storage browser/perangkat yang sama. Menghapus data situs dapat menghapus riwayat tersimpan.

## Menjalankan model Python

Python 3.10+ direkomendasikan. Tidak ada dependensi pihak ketiga.

```bash
python sequence_model.py 3849 8909 8448 9184 4864 5730 5991
python -m unittest discover -s tests -v
python train_text.py
python train_numbers.py
python train_patterns.py
```

Argumen angka harus berupa string empat digit, urut dari lama ke baru. Backtest hanya menguji periode setelah jumlah data minimum tersedia.

## Struktur

```
AI-pola/
├── index.html
├── sequence_model.py
├── tests/
│   ├── test_sequence_model.py
│   └── test_train_models.py
├── datasets/
├── models/
├── train_text.py
├── train_numbers.py
├── train_patterns.py
└── .github/workflows/tests.yml
```

GitHub Actions menjalankan skrip pelatihan dan unit test saat push, pull request, atau pemicu manual.

## Privasi dan lisensi

Riwayat website disimpan lokal pada browser dan tidak dikirim ke server. Repository belum menetapkan lisensi; tambahkan lisensi sebelum penggunaan ulang atau distribusi.
