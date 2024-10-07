import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model
model_fraud = pickle.load(open('model_fraud.sav', 'rb'))

# Load the vocabulary
vocabulary = pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))
loader_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(vocabulary))

# Fit the vectorizer on dummy data to initialize it
loader_vec.fit(["dummy data"])

# Custom CSS to center title and justify text
st.markdown("""
    <style>
    .center-text {
        text-align: center;
    }
    .justify-text {
        text-align: justify;
    }
    .center-table {
        margin-left: auto;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Judul Halaman WEB (Centered)
st.markdown("""
<h2 class="center-text">MENDETEKSI PESAN PENIPUAN PADA SMS</h2>
<h4 class="center-text">Oleh: Muhammad Gilang Dwi Saputra 21.11.4233</h4>
""", unsafe_allow_html=True)

# Menambahkan gambar
st.image('tipu.jpg')

# Dropdown untuk Tentang Aplikasi
section = st.selectbox(
    'Pilih Kategori',
    ('Klik di sini untuk memilih kategori', 'Tentang Aplikasi', 'Cara Penggunaan',
     'Tentang Model', 'Keterangan Input Data')
)

# Konten berdasarkan pilihan dropdown
if section == 'Tentang Aplikasi':
    st.markdown("""
    <div class="justify-text">
    ## Tentang Aplikasi

    Aplikasi web ini bertujuan untuk membantu pengguna mengidentifikasi pesan SMS yang berpotensi merupakan penipuan. 
    Dengan menggunakan model *Naïve Bayes* yang telah dilatih, aplikasi ini dapat menganalisis teks dari pesan SMS 
    dan memberikan prediksi apakah pesan tersebut adalah penipuan atau bukan. 

    Hal ini diharapkan dapat meningkatkan kewaspadaan dan keamanan pengguna terhadap ancaman penipuan melalui SMS.
    </div>
    """, unsafe_allow_html=True)
    
elif section == 'Cara Penggunaan':
    st.markdown("""
    ## Cara Penggunaan

    1. Masukkan teks SMS ke dalam kotak input.
    2. Klik tombol 'Hasil Deteksi' untuk melihat prediksi.
    3. Prediksi akan muncul di bawah tombol.
    """)
    
elif section == 'Tentang Model':
    st.markdown("""
    <div class="justify-text">
    ## Tentang Model

    Model yang digunakan dalam aplikasi ini adalah model *Naïve Bayes* yang telah dilatih menggunakan dataset yang berisi seputar SMS. 
    Model ini menggunakan teknik *TF-IDF* untuk ekstraksi fitur dari teks SMS dan telah diuji untuk memberikan 
    hasil prediksi yang akurat.
    </div>
    """, unsafe_allow_html=True)
    
elif section == 'Keterangan Input Data':
    st.markdown("## Keterangan Input Data")
    data = {
        'Label': [0, 1, 2],
        'Keterangan': ['SMS Normal', 'SMS Penipuan', 'SMS Promo']
    }
    df = pd.DataFrame(data)
    
    # Center the table
    st.markdown("""
    <div class="center-table">
    """, unsafe_allow_html=True)
    st.table(df)
    st.markdown("</div>", unsafe_allow_html=True)

# Input teks SMS
clean_teks = st.text_input('Masukan Teks SMS')

fraud_detection = ''

if st.button('Hasil Deteksi'):
    if clean_teks:  # Pastikan input tidak kosong
        transformed_text = loader_vec.transform([clean_teks])
        predik_fraud = model_fraud.predict(transformed_text)
        predik_proba = model_fraud.predict_proba(transformed_text)

        if predik_fraud == 0:
            fraud_detection = f'SMS Normal dengan akurasi {predik_proba[0][0]*100:.2f}%'
        elif predik_fraud == 1:
            fraud_detection = f'SMS Penipuan dengan akurasi {predik_proba[0][1]*100:.2f}%'
        else:
            fraud_detection = f'SMS Promo dengan akurasi {predik_proba[0][2]*100:.2f}%'

        st.success(fraud_detection)
    else:
        st.warning("Silakan masukkan teks SMS sebelum melakukan prediksi.")
