# halaman_preprocessing.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Fungsi untuk tahapan preprocessing
def show_preprocessing():
    st.title("Tahapan Preprocessing")

    # Memuat data asli
    original_df = pd.read_csv("2_Kategoriberita-CNN.csv")

    st.write("## Data Asli:")
    st.write(original_df[['judul', 'isi', 'tanggal', 'kategori']].head())  # Menampilkan kolom yang relevan dari dataset asli


    # Memuat data
    df = pd.read_csv("preprocessing-cnnnews.csv")

    st.write("## Data yang telah dibersihkan:")
    st.write(df[['berita_clean', 'case_folding', 'tokenize', 'stopword_removal']].head())

    # Menampilkan hasil TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['stopword_removal'])

    st.write("## Hasil TF-IDF:")
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    st.write(tfidf_df.head())

# Menampilkan halaman
if __name__ == "__main__":
    show_preprocessing()
