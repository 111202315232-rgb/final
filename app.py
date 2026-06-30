import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="📊",
    layout="wide"
)

# ── Load model & preprocessor ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model       = joblib.load("best_model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    with open("feature_info.json") as f:
        feature_info = json.load(f)
    return model, preprocessor, feature_info

try:
    model, preprocessor, feature_info = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Gagal memuat model: {e}")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Prediksi Churn Pelanggan")
st.markdown("""
Aplikasi ini memprediksi apakah seorang pelanggan akan **churn** (berhenti menggunakan layanan)
berdasarkan data demografis dan perilaku mereka.

> **Model:** Random Forest (Hyperparameter Tuned)  
> **Dataset:** Sales and Marketing Customer (15.000 records)
""")
st.divider()

# ── Sidebar – penjelasan fitur ────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
**Churn** = pelanggan berhenti menggunakan layanan.

Isi form di bawah dengan data pelanggan, lalu klik **Prediksi** untuk melihat hasilnya.

---
**Label Prediksi:**
- 🟢 **Tidak Churn** – pelanggan kemungkinan tetap aktif
- 🔴 **Churn** – pelanggan berpotensi pergi
""")
    st.header("📌 Deskripsi Fitur Utama")
    st.markdown("""
| Fitur | Keterangan |
|---|---|
| satisfaction_score | Skor kepuasan (0–10) |
| nps_score | Net Promoter Score |
| total_spent | Total pengeluaran |
| lifetime_value | Nilai total pelanggan |
| support_tickets | Jumlah tiket dukungan |
| refund_requested | Pernah minta refund? |
""")

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("📝 Data Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demografi**")
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    age    = st.slider("Usia", 18, 80, 35)
    country = st.selectbox("Negara", ["USA", "UK", "Canada", "Australia", "Germany", "France", "India"])
    city    = st.text_input("Kota", value="New York")

with col2:
    st.markdown("**Aktivitas**")
    subscription_type = st.selectbox("Tipe Langganan", ["Basic", "Standard", "Premium"])
    is_premium_user   = st.radio("Pengguna Premium?", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
    acquisition_channel = st.selectbox("Channel Akuisisi", ["Email", "Ads", "Organic", "Referral", "Social Media"])
    device_type = st.selectbox("Tipe Perangkat", ["Mobile", "Desktop", "Tablet"])
    total_visits = st.number_input("Total Kunjungan", 0, 1000, 50)
    avg_session_time = st.number_input("Avg Session Time (menit)", 0.0, 120.0, 15.0, step=0.5)
    pages_per_session = st.number_input("Halaman per Sesi", 0.0, 50.0, 5.0, step=0.5)

with col3:
    st.markdown("**Transaksi & Kepuasan**")
    total_spent      = st.number_input("Total Pengeluaran ($)", 0.0, 50000.0, 500.0, step=10.0)
    avg_order_value  = st.number_input("Avg Order Value ($)", 0.0, 5000.0, 100.0, step=5.0)
    discount_used    = st.radio("Pernah Pakai Diskon?", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
    support_tickets  = st.slider("Jumlah Support Ticket", 0, 20, 1)
    refund_requested = st.radio("Pernah Minta Refund?", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
    delivery_delay_days = st.slider("Keterlambatan Pengiriman (hari)", 0, 30, 0)
    payment_method   = st.selectbox("Metode Pembayaran", ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"])
    satisfaction_score = st.slider("Skor Kepuasan (0–10)", 0.0, 10.0, 7.0, step=0.1)
    nps_score        = st.slider("NPS Score (-100–100)", -100, 100, 30)

st.markdown("**Pemasaran & Nilai Pelanggan**")
col4, col5 = st.columns(2)
with col4:
    email_open_rate  = st.slider("Email Open Rate (%)", 0.0, 100.0, 30.0, step=0.5)
    email_click_rate = st.slider("Email Click Rate (%)", 0.0, 100.0, 10.0, step=0.5)
    marketing_spend  = st.number_input("Marketing Spend per User ($)", 0.0, 1000.0, 50.0, step=1.0)
with col5:
    lifetime_value   = st.number_input("Lifetime Value ($)", 0.0, 100000.0, 2000.0, step=100.0)
    last_3_month_freq = st.slider("Frekuensi Pembelian 3 Bulan Terakhir", 0, 30, 3)
    coupon_code      = st.text_input("Kode Kupon (kosongkan jika tidak ada)", value="")

# signup_date & last_purchase_date (less critical, pakai default)
signup_date = "2022-01-15"
last_purchase_date = "2024-06-01"

# ── Prediksi ──────────────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔍 Prediksi Churn", type="primary", use_container_width=True)

if predict_btn:
    if not model_loaded:
        st.error("Model belum dimuat. Pastikan file best_model.joblib dan preprocessor.joblib tersedia.")
    else:
        input_data = {
            'customer_id': 0,
            'gender': gender,
            'age': age,
            'country': country,
            'city': city,
            'signup_date': signup_date,
            'last_purchase_date': last_purchase_date,
            'acquisition_channel': acquisition_channel,
            'device_type': device_type,
            'subscription_type': subscription_type,
            'is_premium_user': is_premium_user,
            'total_visits': total_visits,
            'avg_session_time': avg_session_time,
            'pages_per_session': pages_per_session,
            'email_open_rate': email_open_rate,
            'email_click_rate': email_click_rate,
            'total_spent': total_spent,
            'avg_order_value': avg_order_value,
            'discount_used': discount_used,
            'coupon_code': coupon_code if coupon_code else "None",
            'support_tickets': support_tickets,
            'refund_requested': refund_requested,
            'delivery_delay_days': delivery_delay_days,
            'payment_method': payment_method,
            'satisfaction_score': satisfaction_score,
            'nps_score': nps_score,
            'marketing_spend_per_user': marketing_spend,
            'lifetime_value': lifetime_value,
            'last_3_month_purchase_freq': last_3_month_freq,
        }

        df_input = pd.DataFrame([input_data])

        # NOTE: customer_id, signup_date, last_purchase_date TIDAK di-drop di sini.
        # preprocessor.joblib dilatih dengan kolom-kolom ini tetap ada (lihat feature_info.json),
        # jadi harus tetap disertakan saat transform meskipun nilainya dummy/default.

        try:
            X_transformed = preprocessor.transform(df_input)
            prediction    = model.predict(X_transformed)[0]
            proba         = model.predict_proba(X_transformed)[0]

            st.subheader("🎯 Hasil Prediksi")
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                if prediction == 1:
                    st.error("### 🔴 CHURN\nPelanggan ini **berpotensi churn** dan berhenti menggunakan layanan.")
                else:
                    st.success("### 🟢 TIDAK CHURN\nPelanggan ini kemungkinan **tetap aktif** menggunakan layanan.")

            with res_col2:
                st.metric("Probabilitas Tidak Churn", f"{proba[0]*100:.1f}%")
                st.metric("Probabilitas Churn",       f"{proba[1]*100:.1f}%")

            # Bar chart probabilitas
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 2.5))
            bars = ax.barh(["Tidak Churn", "Churn"], [proba[0], proba[1]],
                           color=["#2ecc71", "#e74c3c"])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilitas")
            ax.set_title("Distribusi Probabilitas Prediksi")
            for bar, val in zip(bars, [proba[0], proba[1]]):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{val*100:.1f}%", va='center')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("UAS Bengkel Koding Data Science – Universitas Dian Nuswantoro | Model: Random Forest Tuned")
