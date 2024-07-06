import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Save Model

model_fraud = pickle.load(open('model_fraud.sav', 'rb'))

tfidf = TfidfVectorizer

loader_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(
    pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))

# Judul Halaman WEB

st.title('Prediksi SMS Penipuan')

# Deskripsi singkat aplikasi

st.markdown("""
## Tentang Aplikasi

Aplikasi web ini bertujuan untuk membantu pengguna mengidentifikasi pesan SMS yang berpotensi merupakan penipuan. 
Dengan menggunakan model *Naïve Bayes* yang telah dilatih, aplikasi ini dapat menganalisis teks dari pesan SMS 
dan memberikan prediksi apakah pesan tersebut adalah penipuan atau bukan. 

Hal ini diharapkan dapat meningkatkan kewaspadaan dan keamanan pengguna terhadap ancaman penipuan melalui SMS.
""")

clean_teks = st.text_input('Masukan Teks SMS')


fraud_detection = ''

if st.button('Hasil Detekti'):
    predik_fraud = model_fraud.predict(loader_vec.fit_transform([clean_teks]))

    if (predik_fraud == 0):
        fraud_detection = 'SMS Normal'
    elif (predik_fraud == 1):
        fraud_detection = 'SMS Penipuan'
    else:
        fraud_detection = 'SMS Promo'
st.success(fraud_detection)
