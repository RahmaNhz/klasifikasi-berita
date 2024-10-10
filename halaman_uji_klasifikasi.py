# halaman_uji_klasifikasi.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Fungsi untuk melakukan klasifikasi
def classify_news(news_text):
    # Load model dan TF-IDF Vectorizer yang telah dilatih
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("logreg_model.pkl")

    # Preprocessing input user (misalnya: case folding, stopword removal)
    text = news_text.lower()

    # Transformasi ke TF-IDF
    tfidf_input = vectorizer.transform([text])

    # Prediksi kategori
    prediction = model.predict(tfidf_input)
    return prediction[0]

# Fungsi untuk menampilkan halaman uji coba klasifikasi
def show_classification_page():
    st.title("Uji Coba Klasifikasi Berita")

    # Form input untuk user
    news_text = st.text_area("Masukkan isi berita untuk diklasifikasikan apakah termasuk dalam kategori Berita Ekonomi atau Olahraga")

    if st.button("Klasifikasikan"):
        if news_text:
            prediction = classify_news(news_text)
            st.success(f"Kategori berita: {prediction}")
        else:
            st.warning("Harap masukkan teks berita")

# Menampilkan halaman
if __name__ == "__main__":
    show_classification_page()
