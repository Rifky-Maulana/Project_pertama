# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Permasalahan kesehatan mental, khususnya depresi, semakin menjadi perhatian di kalangan pelajar dan mahasiswa. Depresi pada pelajar dapat berdampak signifikan terhadap performa akademik, hubungan sosial, serta kualitas hidup secara keseluruhan. Oleh karena itu, penting untuk mengidentifikasi faktor-faktor yang dapat memicu atau memperburuk kondisi depresi sejak dini, agar dapat dilakukan intervensi atau pencegahan.

Dataset yang digunakan dalam proyek ini diambil dari platform Kaggle, yang berisi data terkait depresi pada pelajar. Data mencakup informasi demografis (seperti usia dan jenis kelamin), kebiasaan hidup (seperti pola tidur dan aktivitas sosial), riwayat kesehatan mental, serta hasil skoring dari skala depresi standar.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Mengidentifikasi fitur-fitur utama yang berkorelasi dengan tingkat depresi pada pelajar.
- Memprediksi apakah seorang pelajar mengalami depresi berdasarkan informasi yang tersedia menggunakan algoritma machine learning seperti Logistic Regression dan Decision Tree.
- Memberikan insight kepada institusi pendidikan untuk menyusun strategi intervensi yang tepat.
### Referensi:
 * 1] American Psychological Association. (2020). Depression and College Students. APA.
 * [2] WHO. (2021). Depression. https://www.who.int/news-room/fact-sheets/detail/depression

## Business Understanding
## Problem Statement

**Latar Belakang**: Depresi di kalangan pelajar menjadi masalah serius di seluruh dunia. Banyak faktor seperti stres akademik, tekanan keluarga, masalah ekonomi, hingga faktor sosial mempengaruhi tingkat depresi pada siswa. Dengan menganalisis dataset ini, kita bisa memahami faktor-faktor tersebut dan mengambil langkah preventif lebih awal.

Berikut beberapa pernyataan masalah berdasarkan dataset:

*   Apakah tekanan Work/Study Hours dengan Gender memiliki pengaruh signifikan terhadap depresi pelajar?
*   Apakah data dari Family History of Mental Illness memiliki pengaruh terhadap depresi pelajar?
*   Dapatkah model machine learning memprediksi kondisi depresi mahasiswa berdasarkan data numerik seperti tekanan, jam kerja, kepuasan, dan CGPA?


## Goals

*   Analisis korelasi antara Work/Study Hours dengan Gender terhadap label Depression.
*   analisis jumlah pelajar yang depresi dikarenakan pengaruh lingkungan menggunakan Bar Plot atau Count Plot
*   Buat model klasifikasi (target: Depression) menggunakan data numerik tersebut.Evaluasi model dengan F1-score, Confusion Matrix, dan ROC-AUC.

## Solution statements

1. Solusi 1 Baseline Modeling

    -   Gunakan Logistic Regression karena semua fitur numerik → minim preprocessing.
    -   Evaluasi dengan accuracy dan confusion matrix.


2. Solusi 2 Advanced Modeling

    -   Random Forest Classifier → cocok untuk feature importance
    -  XGBoost Classifier → performa tinggi untuk dataset numerik


3. Solusi 3 Hyperparameter Tuning dan Validasi

    -   Gunakan GridSearchCV atau RandomizedSearchCV
    -   Validasi silang (cross-validation) untuk generalisasi model.


## Data Understanding
Dataset yang digunakan dalam proyek ini diambil dari Kaggle: Student Depression Dataset https://www.kaggle.com/datasets/hopesb/student-depression-dataset, yang berisi informasi tentang kondisi mental mahasiswa dan faktor-faktor yang dapat berhubungan dengan depresi. Dataset ini disusun untuk membantu penelitian di bidang psikologi, pendidikan, dan data science, khususnya dalam mendeteksi kemungkinan depresi sejak dini. Dataset ini terdiri dari 27.901 entri dan 18 kolom, yang memuat berbagai informasi mengenai karakteristik demografis, kondisi akademik, dan faktor psikologis mahasiswa.

Dataset ini terdiri dari beberapa fitur/variabel berikut:
### Variabel-variabel pada Student Depression Dataset adalah sebagai berikut:
- id: Nomor identifikasi unik untuk setiap entri.

- Gender: Jenis kelamin responden (misalnya: Male, Female).

- Age: Usia responden.

- City: Kota tempat tinggal atau studi responden.

- Profession: Jenjang atau jurusan pendidikan responden.

- Academic Pressure: Tingkat tekanan akademik yang dirasakan.

- Work Pressure: Tingkat tekanan kerja (jika ada).

- CGPA: Nilai rata-rata akademik responden.

- Study Satisfaction: Tingkat kepuasan terhadap studi.

- Job Satisfaction: Tingkat kepuasan terhadap pekerjaan (jika bekerja).

- Sleep Duration: Lama tidur harian responden (dalam jam).

- Dietary Habits: Pola makan responden (baik/buruk).

- Degree: Jenjang studi saat ini (misalnya: Bachelor's, Master's).

- Have you ever had suicidal thoughts?: Pertanyaan terkait pemikiran bunuh diri (Ya/Tidak).

- Work/Study Hours: Total waktu yang dihabiskan untuk kerja atau belajar per hari.

- Financial Stress: Tingkat tekanan atau beban keuangan.

- Family History of Mental Illness: Riwayat penyakit mental dalam keluarga (Ya/Tidak).

- Depression: Label target untuk klasifikasi, menunjukkan apakah responden mengalami depresi (1 = depresi, 0 = tidak).

**Rubrik/Kriteria Tambahan (Opsional)**:
- dalam menganalisis data pada data Profession saya menemukan tidak hanya data student yang terdia melainkan yang lain juga
  ![image](https://github.com/user-attachments/assets/2e7252df-2e43-481a-b207-3a4b2ede7a96)

  sehingga langkah saya adalah menghapus data colum selain studen karena model ini akan saya gunakan untuk memprediksi seorang pelajar.
  hasil:
  
  ![image](https://github.com/user-attachments/assets/5ee2278c-4226-4949-a998-cc172fb5de3f)

- dalam menganalisi data saya menemukan ada data pada Age yang yaitu sebuah outlier dan tidak relevan dengan umur pelajar

  
  ![image](https://github.com/user-attachments/assets/85310de5-c29e-4c8e-962a-ec8e05858969)


   dan setelah ini saya sudah menghapus outliernya

- lalu selanjutnya saya membuat Headmap korelasi antar variable


![image](https://github.com/user-attachments/assets/fad835c7-1821-43c1-a504-864273ee3f48)





## Data Preparation
Tahapan data preparation dilakukan untuk memastikan data yang digunakan dalam pemodelan bersih, relevan, dan dalam format yang sesuai untuk algoritma machine learning. Adapun langkah-langkah yang dilakukan dalam proses ini adalah sebagai berikut:

1. Pemilihan Fitur
    Fitur-fitur awal yang dipilih berdasarkan analisis korelasi dan relevansi terhadap target (Depression) adalah:

    - Academic Pressure
    
    - Age
    
    - Work/Study Hours
    
    - Study Satisfaction

    Pemilihan fitur ini bertujuan untuk mengurangi dimensi dan fokus pada variabel-variabel yang paling memengaruhi tingkat depresi.

2. Pembuatan Fitur Baru
    Dua fitur baru dibuat untuk mengekstraksi informasi tambahan dari kombinasi variabel yang ada:

    - Pressure_Hours_Ratio = Academic Pressure / (Work/Study Hours + 1)
     → Untuk melihat proporsi tekanan akademik terhadap beban jam kerja/belajar.
    
    - Age_Study_Satisfaction = Age × Study Satisfaction
     → Untuk melihat apakah tingkat kepuasan studi berubah tergantung usia.

3. Pembagian Data
Dataset dibagi menjadi dua bagian:
    
    - Training set: 80% dari data
    
    - Testing set: 20% dari data

    Pembagian ini dilakukan agar model dapat belajar dari sebagian besar data dan dievaluasi secara adil terhadap data yang tidak pernah dilihat sebelumnya.

4. Standardisasi Data
    Fitur numerik yang digunakan dalam model kemudian dilakukan standardisasi menggunakan StandardScaler. Hal ini penting karena algoritma seperti Random Forest dapat dipengaruhi oleh skala fitur jika tidak     dinormalisasi.

**Rubrik/Kriteria Tambahan (Opsional)**: 
#### Proses Data Preparation

1. **Pemilihan Fitur (Feature Selection):**  
   Fitur-fitur yang digunakan dipilih berdasarkan hasil analisis korelasi terhadap target variabel (`Depression`). Fitur-fitur yang terpilih adalah:
   - `Academic Pressure`
   - `Age`
   - `Work/Study Hours`
   - `Study Satisfaction`  
   Fitur-fitur ini memiliki hubungan yang signifikan terhadap tingkat depresi responden.

2. **Pembuatan Fitur Baru (Feature Engineering):**  
   Untuk memperkaya informasi yang diterima model, dua fitur baru dibuat:
   - `Pressure_Hours_Ratio`: Rasio antara tekanan akademik terhadap jam belajar/kerja, untuk mengukur seberapa besar tekanan yang dirasakan dibandingkan waktu belajar.
   - `Age_Study_Satisfaction`: Perkalian antara umur dan kepuasan belajar, untuk melihat apakah usia berpengaruh terhadap kepuasan belajar yang mungkin memengaruhi kondisi psikologis.

3. **Pemisahan Data (Train-Test Split):**  
   Dataset dibagi menjadi dua bagian, yaitu 80% data untuk pelatihan dan 20% untuk pengujian menggunakan fungsi `train_test_split`.  
   Ini dilakukan agar model dapat diuji dengan data yang belum pernah dilihat sebelumnya, sehingga hasil evaluasi lebih objektif.

4. **Standardisasi Data:**  
   Semua fitur diskalakan menggunakan `StandardScaler` agar berada pada skala yang seragam.  
   Meskipun model Random Forest tidak terlalu sensitif terhadap skala data, proses ini tetap dilakukan untuk memastikan kestabilan model dan membantu interpretasi fitur yang mungkin digunakan pada tahap analisis atau visualisasi lanjutan.


#### Alasan Pentingnya Tahapan Data Preparation

- **Menghindari fitur yang tidak relevan atau redundant** agar model tidak terbebani informasi yang tidak penting.
- **Fitur baru membantu model memahami pola kompleks**, yang tidak dapat ditangkap hanya dengan fitur asli.
- **Pemisahan data pelatihan dan pengujian** memastikan bahwa performa model diukur dengan adil dan tidak terjadi overfitting.
- **Standardisasi membantu dalam mempercepat proses pelatihan** dan menjaga kestabilan algoritma, terutama saat fitur memiliki rentang nilai yang sangat berbeda.

## Modeling
Pada tahap ini saya menggunakan algoritma logistic regression, Randomforest, dan DecisionTree

1. **Logistic regression**
   Pada bagian ini, kita akan membahas langkah-langkah yang digunakan untuk melatih dan mengevaluasi model regresi logistik. Model ini digunakan untuk tujuan klasifikasi, di mana kita memprediksi kelas atau kategori dari data berdasarkan fitur yang tersedia.
   - Pelatihan Model (Training the Model)
        Pelatihan model dilakukan dengan menggunakan algoritma Logistic Regression. Langkah pertama adalah menentukan model regresi logistik dan mengonfigurasinya sesuai kebutuhan.

     ![image](https://github.com/user-attachments/assets/1714ca2c-198c-4c7d-91c4-f9ad2e6a233d)

        Pelatihan Model: Model regresi logistik dilatih menggunakan data pelatihan yang telah diskalakan. Parameter max_iter=1000 mengatur jumlah iterasi, dan class_weight='balanced' membantu model mengatasi ketidakseimbangan kelas. Model dipelajari dengan fungsi fit().

1. **Random Forest**
   Secara keseluruhan, kode ini melatih model Random Forest, mengevaluasi kinerjanya, dan kemudian menampilkan hasilnya dengan menggunakan akurasi, laporan klasifikasi, dan matriks kebingungan yang divisualisasikan.

   - Pelatiha Model
         Inisialisasi Model Random Forest: Model Random Forest dibuat dengan parameter yang dioptimalkan. best_params adalah dictionary yang berisi nilai terbaik untuk parameter max_depth, min_samples_split, dan n_estimators.

        - max_depth=10: Menentukan kedalaman maksimum pohon keputusan.

        - min_samples_split=10: Jumlah minimal sampel untuk membagi sebuah node.

        - n_estimators=100: Jumlah pohon keputusan dalam hutan.

        - random_state=42: Untuk memastikan hasil yang konsisten setiap kali kode dijalankan.

        - class_weight='balanced': Menyesuaikan bobot kelas agar model lebih memperhatikan kelas yang jarang muncul.

**Rubrik/Kriteria Tambahan (Opsional)**: 
### Kelebihan dan Kekurangan

**Logistic Regression**

    - Kelebihan:
        - Sederhana, cepat, dan mudah diinterpretasikan.

        - Baik untuk klasifikasi biner dan memberikan probabilitas.

    - Kekurangan:
        - Hanya efektif untuk hubungan linier.

        - Sensitif terhadap outlier dan tidak bisa menangani interaksi fitur tanpa modifikasi.

**Random Forest**

    - Kelebihan:
        - Menangani data non-linier dan interaksi fitur.

        - Robust terhadap overfitting dan data hilang.

    - Kekurangan:
        - Sulit diinterpretasikan.

        - Memakan banyak waktu pelatihan dan memori.
### Hyperparameter tuninhg pada RandomForest
Disini saya menggunakan Grid Search untuk mencari kombinasi parameter terbaik untuk Random Forest dan menyimpannya dalam Best model

![image](https://github.com/user-attachments/assets/3619baba-762c-415a-ac2d-f492fd8b0e1d)

Grid Search mencoba semua kemungkinan kombinasi parameter yang kita tentukan dalam param_grid, lalu mengevaluasi performa model untuk setiap kombinasi menggunakan teknik cross-validation. Tujuannya adalah memilih parameter yang menghasilkan skor evaluasi terbaik (dalam kasus ini, skor F1).

Grid Search akan mencoba semua kombinasi dari nilai-nilai tersebut, misalnya:

    - n_estimators=50, max_depth=None, min_samples_split=2

    - n_estimators=50, max_depth=None, min_samples_split=5

dan seterusnya...

Dengan total 3 × 3 × 3 = 27 kombinasi.

### Pemilihan Model
Disini setelah saya melatih model dengan beberapa algoritma ML saya memutuskan memakai model RandomForest untuk prediksi depresi. Pemilihan Random Forest dalam penelitian ini didasarkan pada beberapa alasan berikut:

- Kemampuan Menangani Data Non-Linier
    Random Forest dapat mempelajari pola yang kompleks dan non-linier antara fitur dan target, yang sangat cocok jika data tidak mengikuti hubungan linier.

- Robust terhadap Overfitting
    Berbeda dengan pohon keputusan tunggal, Random Forest menggunakan banyak pohon dan menggabungkan hasilnya, sehingga mengurangi risiko overfitting dan membuat prediksi lebih stabil.

- Menangani Outlier dan Data Hilang
    Random Forest cukup tangguh terhadap data yang mengandung outlier dan dapat bekerja baik meskipun terdapat nilai yang hilang (missing values).

- Dukungan terhadap Fitur Penting
    Model ini menyediakan nilai feature importance, yang membantu dalam menganalisis fitur mana yang paling berpengaruh terhadap hasil prediksi, sehingga mendukung interpretasi hasil.

- Akurasi Tinggi
    Dalam banyak studi dan praktik, Random Forest dikenal memiliki performa prediksi yang baik untuk berbagai jenis data, terutama pada klasifikasi.

Dengan keunggulan-keunggulan tersebut, Random Forest menjadi pilihan yang tepat untuk membangun model prediksi dalam penelitian ini.

## Evaluation
Saya memilih akurasi (accuracy) sebagai metrik utama karena ingin mengukur seberapa sering model secara keseluruhan memberikan prediksi yang benar. Namun, karena data tidak seimbang (lebih banyak kasus depresi daripada tidak depresi), saya juga menggunakan precision, recall, dan F1-score untuk mengevaluasi performa model secara lebih mendalam. Akurasi (accuracy) adalah salah satu metrik evaluasi yang digunakan untuk mengukur seberapa tepat model klasifikasi dalam membuat prediksi.

![image](https://github.com/user-attachments/assets/0b6fa4cb-49e0-46b0-9d49-49b1e1c2958a)


### Penjelasan Hasil
1. Akurasi (Accuracy = 76.1%)
- Kenapa dipakai?
    Saya menggunakan akurasi karena ingin melihat persentase prediksi benar secara keseluruhan. Nilai 76.1% menunjukkan bahwa model cukup baik dalam memprediksi kedua kelas (depresi dan tidak depresi).

- Kekurangan akurasi:
    Jika data tidak seimbang, akurasi bisa menyesatkan. Misalnya, jika 90% data adalah depresi, model yang selalu prediksi "depresi" akan dapat akurasi 90%, padahal buruk dalam mengenali kasus tidak depresi.
  
![image](https://github.com/user-attachments/assets/95aad483-56ce-4a9d-8029-2ef8e28cbc61)


**Rubrik/Kriteria Tambahan (Opsional)**: 
### Cara Kerja Metrik Accuracy

Metrik **accuracy** bekerja dengan cara membandingkan hasil prediksi model dengan label sebenarnya dari data uji. Setiap kali model membuat prediksi, hasilnya akan dinilai apakah **benar** atau **salah**. Accuracy kemudian menghitung **berapa banyak prediksi yang benar** dari **seluruh prediksi yang dilakukan**.

Berikut langkah-langkahnya dalam bentuk teks:

1. Model melakukan prediksi pada seluruh data uji (X_test).
2. Setiap hasil prediksi dibandingkan dengan label asli (y_test).
3. Jika hasil prediksi sama dengan label asli, maka dianggap **benar**.
4. Setelah semua data diprediksi, hitung jumlah total prediksi yang benar.
5. Accuracy dihitung dengan rumus:


   ![image](https://github.com/user-attachments/assets/ca224711-20d6-45c0-a610-4bc263d8af89)


7. Hasil akhirnya dinyatakan dalam bentuk persen atau desimal (misalnya 0.85 artinya 85%).

Accuracy sangat mudah dihitung dan sering digunakan sebagai metrik dasar untuk mengevaluasi performa model klasifikasi.

**---Ini adalah bagian akhir laporan---**


