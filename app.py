# app.py
import streamlit as st
from halaman_preprocessing import show_preprocessing
from halaman_uji_klasifikasi import show_classification_page

# Sidebar untuk navigasi halaman
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih halaman", ("Preprocessing", "Uji Coba Klasifikasi"))

# Memilih halaman berdasarkan pilihan user
if page == "Preprocessing":
    show_preprocessing()
elif page == "Uji Coba Klasifikasi":
    show_classification_page()
