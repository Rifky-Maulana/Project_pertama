# Laporan Proyek Machine Learning - Rifky Maulana Pasaribu

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

*   Membangun model prediksi kondisi depresi berdasarkan data numerik.
*   Mengetahui hubungan antara jam kerja/belajar dan gender terhadap depresi.
*   Mengukur keandalan model untuk digunakan dalam sistem nyata.

## Solution statements


*   Analisis korelasi antara Work/Study Hours dengan Gender terhadap label Depression.
*   analisis jumlah pelajar yang depresi dikarenakan pengaruh lingkungan menggunakan Bar Plot atau Count Plot
*   Buat model klasifikasi (target: Depression) menggunakan data numerik tersebut.Evaluasi model dengan F1-score, Confusion Matrix, dan ROC-AUC.


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
  
***Dalam Mengenali data penulis tidak menemukan adanya missing value dan data yang duplicate***
  ![image](https://github.com/user-attachments/assets/7f572c1c-5ec7-41b9-9743-ec11dd260cc5)

  ![image](https://github.com/user-attachments/assets/cc8844e7-f63d-4bd6-91ee-45c78a9e450c)



**Rubrik/Kriteria Tambahan (Opsional)**:
- dalam menganalisis data pada data Profession penulis menemukan tidak hanya data student yang terdia melainkan yang lain juga
  ![image](https://github.com/user-attachments/assets/2e7252df-2e43-481a-b207-3a4b2ede7a96)

  sehingga langkah penulis adalah menghapus data colum selain studen karena model ini akan digunakan untuk memprediksi seorang pelajar.
  hasil:
  
  ![image](https://github.com/user-attachments/assets/5ee2278c-4226-4949-a998-cc172fb5de3f)

- dalam menganalisi data penulis menemukan ada data pada Age yang yaitu sebuah outlier dan tidak relevan dengan umur pelajar dengan teknik visualisasi boxplot

  
  ![image](https://github.com/user-attachments/assets/85310de5-c29e-4c8e-962a-ec8e05858969)


   dan setelah ini penulis menghapus data selain student karena tidak relevan dengan tema project

- lalu selanjutnya penulis membuat Headmap korelasi antar variable


![image](https://github.com/user-attachments/assets/fad835c7-1821-43c1-a504-864273ee3f48)





## Data Preparation
Tahapan data preparation dilakukan untuk memastikan data yang digunakan dalam pemodelan bersih, relevan, dan dalam format yang sesuai untuk algoritma machine learning. Adapun langkah-langkah yang dilakukan dalam proses ini adalah sebagai berikut:

1. Penanganan Outlier

   ![image](https://github.com/user-attachments/assets/8532ed5c-3386-44df-b983-564140e8d47b)

Disini Penulis menggunakan tekni IQR, IQR adalah singkatan dari Interquartile Range atau Rentang Antarkuartil dalam bahasa Indonesia. QR adalah ukuran statistik yang menunjukkan sebaran tengah dari data, yaitu rentang antara kuartil ketiga (Q3) dan kuartil pertama (Q1). Lalu penulis menerapkan teknik ini untuk membuat rentang pada kolom 'Age'

Lalu untuk penanganan missing value, duplicate dll itu tidak diperlukan karena tidak ditemukan hal tersebut dalam data

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
   Feature selection adalah proses memilih fitur-fitur yang paling relevan dari seluruh data yang tersedia untuk digunakan dalam proses pemodelan. Proses ini sangat penting karena dapat meningkatkan kinerja         model secara keseluruhan. Dengan menghilangkan fitur yang tidak relevan atau yang mengandung banyak noise, model akan lebih fokus mempelajari pola yang benar-benar penting, sehingga hasil prediksi menjadi        lebih akurat. Selain itu, feature selection membantu mengurangi risiko overfitting, yaitu kondisi ketika model terlalu menyesuaikan diri dengan data latih sehingga performanya menurun saat diuji pada data        baru. Dari sisi efisiensi, model dengan fitur yang lebih sedikit akan membutuhkan waktu pelatihan dan komputasi yang lebih cepat.
   Fitur-fitur yang digunakan dipilih berdasarkan hasil analisis korelasi terhadap target variabel (`Depression`). Fitur-fitur yang terpilih adalah:
   - `Academic Pressure`
   - `Age`
   - `Work/Study Hours`
   - `Study Satisfaction`  
   Fitur-fitur ini memiliki hubungan yang signifikan terhadap tingkat depresi responden. 

2. **Pembuatan Fitur Baru (Feature Engineering):**
   Feature engineering adalah proses membuat, mengubah, atau memilih fitur (variabel) dari data mentah agar model machine learning dapat bekerja lebih baik. Proses ini sangat diperlukan karena kualitas fitur sangat menentukan kualitas prediksi yang dihasilkan oleh model. Alasan utama kenapa feature engineering penting adalah karena data mentah sering kali belum cukup representatif untuk langsung digunakan.
   Untuk memperkaya informasi yang diterima model, dua fitur baru dibuat:
   - `Pressure_Hours_Ratio`: Rasio antara tekanan akademik terhadap jam belajar/kerja, untuk mengukur seberapa besar tekanan yang dirasakan dibandingkan waktu belajar.
   - `Age_Study_Satisfaction`: Perkalian antara umur dan kepuasan belajar, untuk melihat apakah usia berpengaruh terhadap kepuasan belajar yang mungkin memengaruhi kondisi psikologis.

3. **Pemisahan Data (Train-Test Split):**
   Train-test split adalah proses membagi dataset menjadi dua bagian utama: data latih (train) dan data uji (test). Proses ini sangat penting dalam machine learning karena bertujuan untuk mengevaluasi performa model secara objektif.Ketika kita melatih model, kita ingin model tersebut mampu mempelajari pola dari data latih, lalu menguji kemampuannya pada data yang belum pernah dilihat sebelumnya, yaitu data uji. Jika kita hanya melatih dan menguji model pada data yang sama, maka model bisa saja terlihat sangat akurat padahal sebenarnya hanya "menghafal" data (terjadi overfitting). Dengan adanya pemisahan train-test, kita bisa melihat apakah model benar-benar bisa menggeneralisasi dan bekerja baik terhadap data baru.
   Dataset dibagi menjadi dua bagian, yaitu 80% data untuk pelatihan dan 20% untuk pengujian menggunakan fungsi `train_test_split`.  
   Ini dilakukan agar model dapat diuji dengan data yang belum pernah dilihat sebelumnya, sehingga hasil evaluasi lebih objektif.

4. **Standardisasi Data:**  
   Semua fitur diskalakan menggunakan `StandardScaler` agar berada pada skala yang seragam. Standardisasi data adalah proses transformasi fitur numerik agar memiliki rata-rata (mean) 0 dan standar deviasi 1. Tujuannya adalah untuk menyamakan skala semua fitur sehingga tidak ada satu fitur pun yang mendominasi pembelajaran model hanya karena memiliki rentang nilai yang lebih besar. 
   Meskipun model Random Forest tidak terlalu sensitif terhadap skala data, proses ini tetap dilakukan untuk memastikan kestabilan model dan membantu interpretasi fitur yang mungkin digunakan pada tahap analisis atau visualisasi lanjutan.
   Proses standardisasi biasanya dilakukan dengan rumus:

   ![image](https://github.com/user-attachments/assets/3d5d5c7b-16d6-4779-8ffa-6494a04d1568)



#### Alasan Pentingnya Tahapan Data Preparation

- **Menghindari fitur yang tidak relevan atau redundant** agar model tidak terbebani informasi yang tidak penting.
- **Fitur baru membantu model memahami pola kompleks**, yang tidak dapat ditangkap hanya dengan fitur asli.
- **Pemisahan data pelatihan dan pengujian** memastikan bahwa performa model diukur dengan adil dan tidak terjadi overfitting.
- **Standardisasi membantu dalam mempercepat proses pelatihan** dan menjaga kestabilan algoritma, terutama saat fitur memiliki rentang nilai yang sangat berbeda.

## Modeling
Pada tahap ini penulis menggunakan algoritma logistic regression, Randomforest, dan DecisionTree

1. **Logistic regression**
   Pada bagian ini, penulis membahas langkah-langkah yang digunakan untuk melatih dan mengevaluasi model regresi logistik. Model ini digunakan untuk tujuan klasifikasi, di mana kita memprediksi kelas atau kategori dari data berdasarkan fitur yang tersedia.
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
Disini penulis menggunakan Grid Search untuk mencari kombinasi parameter terbaik untuk Random Forest dan menyimpannya dalam Best model

![image](https://github.com/user-attachments/assets/3619baba-762c-415a-ac2d-f492fd8b0e1d)

Grid Search mencoba semua kemungkinan kombinasi parameter yang kita tentukan dalam param_grid, lalu mengevaluasi performa model untuk setiap kombinasi menggunakan teknik cross-validation. Tujuannya adalah memilih parameter yang menghasilkan skor evaluasi terbaik (dalam kasus ini, skor F1).

Grid Search akan mencoba semua kombinasi dari nilai-nilai tersebut, misalnya:

    - n_estimators=50, max_depth=None, min_samples_split=2

    - n_estimators=50, max_depth=None, min_samples_split=5

dan seterusnya...

Dengan total 3 × 3 × 3 = 27 kombinasi.

### Pemilihan Model
Disini setelah penulis melatih model dengan beberapa algoritma ML penulis memutuskan memakai model RandomForest untuk prediksi depresi. Pemilihan Random Forest dalam penelitian ini didasarkan pada beberapa alasan berikut:

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
Penulis memilih akurasi (accuracy) sebagai metrik utama karena ingin mengukur seberapa sering model secara keseluruhan memberikan prediksi yang benar. Namun, karena data tidak seimbang (lebih banyak kasus depresi daripada tidak depresi), penulis juga menggunakan precision, recall, dan F1-score untuk mengevaluasi performa model secara lebih mendalam. Akurasi (accuracy) adalah salah satu metrik evaluasi yang digunakan untuk mengukur seberapa tepat model klasifikasi dalam membuat prediksi.

![image](https://github.com/user-attachments/assets/0b6fa4cb-49e0-46b0-9d49-49b1e1c2958a)

###   1. Apakah tekanan Work/Study Hours dengan Gender memiliki pengaruh signifikan terhadap depresi pelajar?
Berdasarkan grafik yang menampilkan hubungan antara durasi kerja/studi harian dengan gejala depresi pada kelompok gender berbeda, terlihat pola yang cukup jelas. Pada siswa laki-laki, gejala depresi mulai muncul ketika durasi kerja/studi mencapai 6 jam per hari dan semakin intens seiring bertambahnya waktu, dengan puncak yang jelas terlihat pada durasi 9 jam dan 12 jam per hari. Pola ini menunjukkan bahwa beban kerja atau studi yang terlalu panjang, terutama yang melebihi 6 jam sehari, secara signifikan berkorelasi dengan munculnya gejala depresi pada populasi laki-laki. Durasi ekstrem hingga 12 jam tampaknya memberikan dampak yang paling berat, mungkin karena kombinasi faktor kelelahan fisik, mental, dan kurangnya waktu untuk pemulihan atau aktivitas lainnya. Berdasarkan grafik yang ditampilkan, analisis ini bersifat univariat (analisis satu variabel) untuk masing-masing kelompok gender, bukan multivariat.

![image](https://github.com/user-attachments/assets/21522fd1-e9a1-44cb-91db-5c66698932ec)

Grafik ini menunjukkan analisis univariat sederhana tentang hubungan durasi kerja/studi dengan depresi, yang dipisahkan berdasarkan gender. Untuk memahami hubungan yang lebih kompleks, diperlukan pendekatan multivariat dengan memasukkan variabel-variabel tambahan.


###    2. Apakah data dari Family History of Mental Illness memiliki pengaruh terhadap depresi pelajar?

![image](https://github.com/user-attachments/assets/7dc43458-cfff-478c-bef5-36d06c37298d)

Data ini menunjukkan hubungan antara riwayat penyakit mental keluarga (Family History of Mental Illness) dengan status depresi (Depression) pada individu. Terdapat empat kategori data utama: (1) Tidak ada riwayat keluarga dan tidak depresi (No 0) sebanyak 6.330 kasus, (2) Tidak ada riwayat keluarga tetapi depresi (No 1) sebanyak 8.049 kasus, (3) Ada riwayat keluarga tetapi tidak depresi (Yes 0) sebanyak 5.222 kasus, dan (4) Ada riwayat keluarga dan depresi (Yes 1) sebanyak 8.257 kasus. 

Dari data tersebut terlihat bahwa kelompok dengan riwayat keluarga penyakit mental memiliki jumlah kasus depresi sedikit lebih tinggi (8.257) dibandingkan kelompok tanpa riwayat (8.049). Namun, kelompok tanpa riwayat keluarga justru menunjukkan jumlah yang lebih besar pada individu yang tidak mengalami depresi (6.330 berbanding 5.222 pada kelompok dengan riwayat). Data ini bersifat **univariate** karena hanya menganalisis hubungan antara satu variabel independen (riwayat penyakit mental keluarga) dengan satu variabel dependen (status depresi), tanpa mempertimbangkan variabel-variabel lain yang mungkin berpengaruh. Analisis ini memberikan gambaran dasar tentang hubungan antara dua variabel tersebut tanpa melihat interaksi dengan faktor-faktor lain.

###    3. Dapatkah model machine learning memprediksi kondisi depresi mahasiswa berdasarkan data numerik seperti tekanan, jam kerja, kepuasan, dan CGPA?

Evaluasi performa model menunjukkan hasil yang cukup baik dengan akurasi sebesar 0.77 (77%). Analisis lebih mendalam melalui classification report mengungkapkan bahwa model memiliki precision 0.82 dan recall 0.77 untuk kelas 1 (depresi), serta precision 0.70 dan recall 0.76 untuk kelas 0 (tidak depresi). Nilai f1-score yang seimbang di kedua kelas (0.79 untuk kelas 1 dan 0.73 untuk kelas 0) menunjukkan bahwa model dapat memprediksi kedua kelas dengan cukup baik tanpa bias yang signifikan.

![image](https://github.com/user-attachments/assets/d97fbfe8-29b0-4847-bcaf-c36ecef48d3e)

Hasil ini mengindikasikan bahwa tekanan akademik dan jam kerja/studi yang panjang merupakan faktor risiko utama depresi, sementara usia yang lebih tua dan kepuasan terhadap studi berperan sebagai faktor protektif. Model ini berhasil mengidentifikasi pola penting dalam data dengan performa yang cukup baik untuk digunakan sebagai dasar pengambilan keputusan terkait kesehatan mental mahasiswa.


### Penjelasan Hasil
1. Akurasi (Accuracy = 76.1%)
- Kenapa dipakai?
    Penulis menggunakan akurasi karena ingin melihat persentase prediksi benar secara keseluruhan. Nilai 76.1% menunjukkan bahwa model cukup baik dalam memprediksi kedua kelas (depresi dan tidak depresi).

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

