# Supervised Learning
Algoritma supervised learning berupa Logistic Regression, KNN, dan ID3. Ketiga algoritma tersebut dibuat dalam bahasa Python dengan bantuan library NumPy.

## Algoritma
### Logistic Regression
Logistic Regression yang diimplementasikan mampu membuat boundary antar dua label data dengan fungsi sigmoid melalui data yang diberikan.

Parameter yang dapat diatur:
- learning rate. Default=0.1.
- epochs (jumlah iterasi). Default=100.

### K-Nearest Neighbors (KNN)
KNN menyimpan data latih dan menggunakannya untuk memprediksi data tes dengan acuan sebanyak K tetangga terdekat yang berlabel. Data tes akan diberi label berdasarkan label mayoritas dari K tetangga terdekat.

Parameter yang dapat diatur:
- K. Default=5.

### Decision Tree ID3
Algoritma decision tree dengan acuan ID3. Algoritma ini membentuk dictionary Python yang rekursif sebagai pohon untuk penentuan keputusan.

## Cara penggunaan

Install Python 3 terlebih dahulu. Python 3.10.4 direkomendasikan.

Library tambahan:
- NumPy
- Pandas

Gunakan perintah berikut untuk menjalankan program

    python main.py [args..]

Argumen yang dapat digunakan

- **-m, --model**: Model yang digunakan (id_3, knn, atau log_reg).
- **-d, --data**: Lokasi data latih yang digunakan.
- **-v, --validation**: Lokasi data validasi untuk mendapatkan akurasi validasi dari model.
- **-t, --test**: Lokasi data tes untuk diprediksi.
- **--lr**: Learning rate. Khusus untuk Logistic Regression. Diabaikan apabila menggunakan model lain.
- **--epochs**: Epochs. Khusus untuk Logistic Regression. Diabaikan apabila menggunakan model lain.
- **--k_nearest**: Nilai K. Khusus untuk KNN. Diabaikan apabila menggunakan model lain.

## Data

### Training Data
Diperlukan nama kolom serta binary label yang diprediksi pada kolom terakhir. Formatnya adalah .csv.

### Validation Data
Sama seperti training data

### Testing Data
Sama seperti training data tetapi tidak boleh ada kolom label.


# Tentang
Repo ini dibuat untuk seleksi Ca-GaIB 2022