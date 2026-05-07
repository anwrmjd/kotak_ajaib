"""
app.py — Aplikasi Prediksi Berbasis Model Orange Data Mining
Dijalankan di Streamlit Cloud.
Semua file (app.py, model_orange.pickle, requirements.txt) harus
berada di GitHub repository yang sama.

Pastikan nama fitur di FEATURE_CONFIG sesuai dengan nama kolom/variabel
saat training di Orange Data Mining.
"""

import os
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────
# KONFIGURASI PATH MODEL
# Gunakan path relatif agar kompatibel dengan Streamlit Cloud.
# Jangan gunakan path absolut seperti /content/... atau C:/Users/...
# ──────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model_orange.pickle"

# ──────────────────────────────────────────────
# KONFIGURASI FITUR
# ⚠️  PENTING: Sesuaikan nama kunci (key) di bawah ini dengan nama
#              kolom/variabel saat training di Orange Data Mining.
#              Urutan kunci menentukan urutan kolom DataFrame input.
# ──────────────────────────────────────────────
FEATURE_CONFIG = {
    # ── Fitur Numerik ──────────────────────────────────────────────
    "umur": {
        "type": "numeric",
        "input": "slider",       # "slider" | "number"
        "label": "Umur (tahun)",
        "min": 0,
        "max": 100,
        "default": 30,
        "step": 1,
    },
    "pendapatan": {
        "type": "numeric",
        "input": "number",
        "label": "Pendapatan (Rp)",
        "min": 0,
        "max": 100_000_000,
        "default": 5_000_000,
        "step": 500_000,
    },
    "lama_bekerja": {
        "type": "numeric",
        "input": "slider",
        "label": "Lama Bekerja (tahun)",
        "min": 0,
        "max": 40,
        "default": 5,
        "step": 1,
    },
    # ── Fitur Kategorikal ──────────────────────────────────────────
    "jenis_kelamin": {
        "type": "categorical",
        "label": "Jenis Kelamin",
        "options": ["Laki-laki", "Perempuan"],
    },
    "status_pernikahan": {
        "type": "categorical",
        "label": "Status Pernikahan",
        "options": ["Belum Menikah", "Menikah", "Cerai"],
    },
}

# ──────────────────────────────────────────────
# LOAD MODEL (di-cache agar tidak reload setiap interaksi)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model…")
def load_model():
    """
    Memuat model dari file pickle.
    Mengembalikan (model, error_message).
    Jika berhasil, error_message = None.
    """
    if not MODEL_PATH.exists():
        return None, (
            f"File model tidak ditemukan: `{MODEL_PATH}`\n\n"
            "Pastikan file `model_orange.pickle` sudah di-upload ke "
            "GitHub repository yang sama dengan `app.py`."
        )
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as exc:
        return None, (
            f"Model gagal dimuat: {exc}\n\n"
            "Periksa apakah file pickle tidak korup dan kompatibel "
            "dengan versi library yang terinstal."
        )


# ──────────────────────────────────────────────
# BUAT FORM INPUT
# ──────────────────────────────────────────────
def create_input_form() -> dict | None:
    """
    Menampilkan form input berdasarkan FEATURE_CONFIG.
    Mengembalikan dict {nama_fitur: nilai} saat tombol ditekan,
    atau None jika form belum disubmit.
    """
    with st.form("prediction_form"):
        st.subheader("📝 Input Data")
        input_data: dict = {}

        cols = st.columns(2)
        col_idx = 0

        for feature_name, cfg in FEATURE_CONFIG.items():
            label = cfg.get("label", feature_name)
            container = cols[col_idx % 2]

            if cfg["type"] == "numeric":
                if cfg.get("input") == "slider":
                    val = container.slider(
                        label,
                        min_value=cfg["min"],
                        max_value=cfg["max"],
                        value=cfg["default"],
                        step=cfg.get("step", 1),
                    )
                else:  # "number"
                    val = container.number_input(
                        label,
                        min_value=float(cfg["min"]),
                        max_value=float(cfg["max"]),
                        value=float(cfg["default"]),
                        step=float(cfg.get("step", 1)),
                    )
            elif cfg["type"] == "categorical":
                val = container.selectbox(label, options=cfg["options"])
            else:
                val = container.text_input(label)

            input_data[feature_name] = val
            col_idx += 1

        submitted = st.form_submit_button(
            "🔍 Prediksi",
            use_container_width=True,
            type="primary",
        )

    return input_data if submitted else None


# ──────────────────────────────────────────────
# PREDIKSI — Pendekatan Scikit-learn-like
# ──────────────────────────────────────────────
def predict_with_model(model, input_df: pd.DataFrame):
    """
    Mencoba prediksi menggunakan antarmuka scikit-learn standar.
    Mengembalikan (prediction, proba_df | None).
    """
    prediction = model.predict(input_df)

    proba_df = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(input_df)
            classes = (
                model.classes_
                if hasattr(model, "classes_")
                else [f"Kelas {i}" for i in range(proba.shape[1])]
            )
            proba_df = pd.DataFrame(
                proba, columns=[str(c) for c in classes]
            )
        except Exception:
            pass

    return prediction, proba_df


# ──────────────────────────────────────────────
# PREDIKSI — Fallback Orange3
# ──────────────────────────────────────────────
def predict_with_orange_fallback(model, input_df: pd.DataFrame):
    """
    Fallback: konversi DataFrame ke format Orange.data.Table
    lalu jalankan prediksi.
    Mengembalikan (prediction, proba_df | None).
    """
    try:
        import Orange.data as od
    except ImportError:
        raise ImportError(
            "Library `orange3` tidak tersedia. "
            "Tambahkan `orange3` ke `requirements.txt`."
        )

    # Bangun domain Orange
    domain_vars = []
    for feature_name, cfg in FEATURE_CONFIG.items():
        if cfg["type"] == "numeric":
            domain_vars.append(od.ContinuousVariable(feature_name))
        elif cfg["type"] == "categorical":
            domain_vars.append(
                od.DiscreteVariable(feature_name, values=cfg["options"])
            )
        else:
            domain_vars.append(od.StringVariable(feature_name))

    domain = od.Domain(domain_vars)

    # Bangun array numpy dari input_df
    row = []
    for feature_name, cfg in FEATURE_CONFIG.items():
        val = input_df[feature_name].iloc[0]
        if cfg["type"] == "categorical":
            options = cfg["options"]
            row.append(float(options.index(val)))
        else:
            row.append(float(val))

    X = np.array([row], dtype=float)
    orange_table = od.Table.from_numpy(domain, X)

    result = model(orange_table)

    # Ekstrak nilai prediksi
    try:
        prediction = [domain.class_var.values[int(r)] for r in result]
    except Exception:
        prediction = list(result)

    # Probabilitas (jika tersedia)
    proba_df = None
    try:
        proba = model(orange_table, model.Probs)
        class_vals = domain.class_var.values
        proba_df = pd.DataFrame(
            proba, columns=[str(v) for v in class_vals]
        )
    except Exception:
        pass

    return prediction, proba_df


# ──────────────────────────────────────────────
# TAMPILKAN HASIL PREDIKSI
# ──────────────────────────────────────────────
def display_result(prediction, proba_df, input_df: pd.DataFrame):
    st.markdown("---")
    st.subheader("📊 Hasil Prediksi")

    # Tabel input
    with st.expander("🔎 Data Input yang Digunakan", expanded=True):
        display_df = input_df.copy()
        display_df.index = ["Input Anda"]
        st.dataframe(display_df, use_container_width=True)

    # Hasil prediksi
    pred_value = prediction[0] if hasattr(prediction, "__len__") else prediction
    st.success(f"✅ **Hasil Prediksi: {pred_value}**")

    # Probabilitas / confidence
    if proba_df is not None:
        st.markdown("#### 📈 Probabilitas per Kelas")
        # Bar chart
        chart_data = proba_df.T.rename(columns={0: "Probabilitas"})
        st.bar_chart(chart_data)
        # Tabel
        st.dataframe(
            proba_df.style.format("{:.2%}").highlight_max(axis=1),
            use_container_width=True,
        )


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────
def main():
    # ── Konfigurasi Halaman ────────────────────────────────────────
    st.set_page_config(
        page_title="Prediksi Model Orange",
        page_icon="🍊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/"
            "Orange_Data_Mining_Logo.png/640px-Orange_Data_Mining_Logo.png",
            width=140,
        )
        st.markdown("## 🍊 Panduan Penggunaan")
        st.markdown(
            """
            1. **Isi semua kolom** input di halaman utama.
            2. Klik tombol **🔍 Prediksi**.
            3. Hasil prediksi akan muncul di bawah form.

            ---
            **ℹ️ Info Model**

            Model dimuat dari file `model_orange.pickle`
            yang tersimpan di GitHub repository yang sama
            dengan aplikasi ini.

            Jika file model berukuran besar (>100 MB),
            pertimbangkan menggunakan **Git LFS** atau
            simpan di external storage (Google Drive, S3, dll.)
            lalu muat menggunakan `requests` / `gdown`.

            ---
            **📁 Struktur Repository**
            ```
            repo/
            ├── app.py
            ├── model_orange.pickle
            ├── requirements.txt
            └── .streamlit/
                └── config.toml
            ```
            """
        )
        st.markdown("---")
        st.caption("Dibuat dengan Streamlit · Ditenagai Orange Data Mining")

    # ── Header Utama ───────────────────────────────────────────────
    st.title("🍊 Aplikasi Prediksi Berbasis Model Orange")
    st.markdown(
        "Aplikasi ini menggunakan model *machine learning* hasil training "
        "dari **Orange Data Mining** dan dijalankan melalui **Streamlit Cloud**."
    )
    st.markdown("---")

    # ── Load Model ─────────────────────────────────────────────────
    model, model_error = load_model()

    if model_error:
        st.error(f"❌ **Model Gagal Dimuat**\n\n{model_error}")
        st.stop()

    st.info(
        f"✅ Model berhasil dimuat dari `{MODEL_PATH.name}` — "
        f"Tipe model: `{type(model).__name__}`"
    )

    # ── Form Input ─────────────────────────────────────────────────
    input_data = create_input_form()

    # ── Prediksi ───────────────────────────────────────────────────
    if input_data is not None:
        input_df = pd.DataFrame([input_data], columns=list(FEATURE_CONFIG.keys()))

        with st.spinner("Memproses prediksi…"):
            prediction, proba_df = None, None
            error_log = []

            # Coba 1: scikit-learn-like
            try:
                prediction, proba_df = predict_with_model(model, input_df)
            except Exception as e1:
                error_log.append(("scikit-learn-like", str(e1)))

                # Coba 2: Orange fallback
                try:
                    prediction, proba_df = predict_with_orange_fallback(
                        model, input_df
                    )
                except ImportError as ie:
                    error_log.append(("orange3-import", str(ie)))
                    st.error(
                        "❌ **Prediksi Gagal — Library orange3 Tidak Tersedia**\n\n"
                        f"{ie}\n\nTambahkan `orange3` ke `requirements.txt` "
                        "dan redeploy aplikasi."
                    )
                    st.stop()
                except Exception as e2:
                    error_log.append(("orange-fallback", str(e2)))

        if prediction is not None:
            display_result(prediction, proba_df, input_df)
        else:
            st.error(
                "❌ **Prediksi Gagal**\n\n"
                "Semua pendekatan prediksi mengalami error. "
                "Detail error ditampilkan di bawah:"
            )
            with st.expander("🐛 Detail Error (untuk debugging)"):
                for method, err in error_log:
                    st.markdown(f"**Metode `{method}`:**")
                    st.code(err, language="text")
                st.markdown(
                    "**Kemungkinan penyebab:**\n"
                    "- Nama kolom di `FEATURE_CONFIG` tidak sesuai dengan "
                    "nama variabel saat training di Orange.\n"
                    "- Format model Orange berbeda dari yang didukung.\n"
                    "- Tipe data input tidak sesuai.\n\n"
                    "Sesuaikan `FEATURE_CONFIG` di `app.py` dengan "
                    "skema data training."
                )


if __name__ == "__main__":
    main()
