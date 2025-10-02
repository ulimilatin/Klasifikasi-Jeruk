import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Klasifikasi Jeruk",
    page_icon="🍊"
)

# Load model
model = joblib.load("model_klasifikasi_jeruk.joblib")

st.title("🍊 Klasifikasi Jeruk")
st.markdown("Aplikasi Machine Learning untuk *Klasifikasi Jeruk*")

diameter = st.slider("Diameter", 0.0, 10.0, 5.0, step=0.1)
berat = st.slider("Berat", 0.0, 300.0, 120.0, step=0.1)
tebal = st.slider("Tebal Kulit", 0.0, 2.0, 0.45, step=0.1)
kadar = st.slider("Kadar Gula", 0.0, 15.0, 5.0, step=0.1)
asal = st.pills("Asal Daerah", ["Jawa Tengah", "Jawa Barat", "Kalimantan"], default="Kalimantan")
warna = st.pills("Warna", ["hijau", "kuning", "oranye"], default="hijau")
musim = st.pills("Musim Panen", ["hujan", "kemarau"], default="hujan")

if st.button("Prediksi", type="primary"):
    data_baru = pd.DataFrame([[diameter, berat, tebal, kadar, asal, warna, musim]],
                         columns=["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","warna","musim_panen"])
    prediksi = model.predict(data_baru)[0]
    presentase = max(model.predict_proba(data_baru)[0])
    st.success(f"Prediksi {prediksi} dengan tingkat keyakinan {presentase*100:.2f}%")
    st.balloons()

st.divider()
st.caption("Dibuat dengan 🤡 oleh Fadwa Pamulasih")