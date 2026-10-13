# Ai Pola

**Ai Pola** adalah proyek pembelajaran machine learning untuk mengeksplorasi pola data. Versi ini menambahkan alur khusus untuk riwayat angka 4 digit, kandidat berikutnya, dan pengujian historis.

## Fitur

- **Analisis urutan 4 digit di website** — masukkan data dari yang paling lama ke paling baru; nol di depan tetap dipertahankan.
- **Kandidat tunggal** — baseline frekuensi digit per posisi menghasilkan tepat satu kandidat jika tersedia minimal 5 data.
- **Pencatatan harian** — tambahkan satu hasil terbaru, periksa apakah kandidat sebelumnya cocok, lalu perbarui kandidat.
- **Backtest walk-forward** — tiap prediksi historis hanya memakai data yang sudah tersedia pada saat itu.
- **Ringkasan digit** — tampilkan frekuensi digit pada setiap posisi.
- **Penyimpanan lokal** — riwayat website disimpan di browser/perangkat yang digunakan; tidak dikirim ke server.
- **Model edukasi lain** — skrip klasifikasi teks dan pengelompokan pola tetap tersedia.

## Cara kerja kandidat

Pada masing-masing dari empat posisi, program memilih digit yang paling sering muncul di posisi tersebut. Jika beberapa digit seri, program memilih digit seri yang paling baru muncul. Keempat digit digabungkan menjadi satu kandidat.

Ini adalah baseline sederhana yang transparan, bukan model yang terbukti mampu meramalkan hasil acak. Akurasi backtest dilaporkan sebagai kecocokan tepat seluruh 4 digit. Hasil historis tidak menjamin hasil berikutnya; jangan menggunakannya sebagai jaminan atau dasar taruhan.

## Menggunakan website

Buka [website melalui GitHub Pages](https://thatosenyum95-ship-it.github.io/AI-pola/) jika GitHub Pages telah diaktifkan pada repository. Jika belum aktif, buka `index.html` dari repository atau aktifkan Pages di Settings → Pages.

1. Riwayat awal berisi contoh angka yang diberikan untuk memulai.
2. Pastikan urutannya dari lama ke baru, lalu tekan **Gunakan riwayat** setelah mengedit.
3. Setiap kali memperoleh hasil baru, masukkan satu angka tepat 4 digit ke bagian **Tambah hasil terbaru**.
4. Baca kandidat dan evaluasi historis. Simpanan website berada di browser yang sama.

## Menjalankan model Python

Program memakai Python standard library:

```bash
python sequence_model.py 3849 8909 8448 9184 4864 5730 5991
python -m unittest discover -s tests -v
python train_text.py
python train_numbers.py
python train_patterns.py
```

Argumen angka harus berupa string 4 digit dan berurutan dari lama ke baru. Skrip Python memvalidasi format, menghasilkan satu kandidat, dan menjalankan backtest walk-forward.

## Struktur

```
AI-pola/
├── index.html
├── sequence_model.py
├── tests/
│   └── test_sequence_model.py
├── datasets/
├── models/
├── train_text.py
├── train_numbers.py
├── train_patterns.py
└── .github/workflows/tests.yml
```

GitHub Actions menjalankan skrip model dan tes saat ada push atau pull request. Periksa tab **Actions** untuk status eksekusi terbaru.

## Privasi dan batasan

Website menyimpan riwayat di local storage browser. Menghapus data situs/browser dapat menghapus riwayat tersimpan. Jangan masukkan informasi pribadi atau rahasia ke repository publik.

## Lisensi

Belum ditetapkan. Tentukan lisensi sebelum membagikan atau menggunakan ulang proyek ini.
