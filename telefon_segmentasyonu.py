import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image


MODEL_DOSYASI = "turk.pt"       
SABIT_CONFIDENCE = 0.25         

st.set_page_config(page_title="Pasaj AI Eksper", page_icon="📱", layout="wide")


st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 10%, #ffc900 0%, #ffffff 50%, #eef2f3 100%);
        background-attachment: fixed;
    }
    h1, h2, h3, h4, h5, h6, p, span, div, label { color: #2c3e50 !important; }
    
    .stButton > button { 
        background-color: #2c3e50; 
        border-radius: 12px; 
        font-weight: bold; 
        width: 100%; 
        height: 55px; 
        border: 2px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button p { color: #ffffff !important; font-size: 18px; }
    .stButton > button:hover { 
        background-color: #ffc900; 
        border: 2px solid #2c3e50; 
        transform: scale(1.02);
    }
    .stButton > button:hover p { color: #2c3e50 !important; }

    div[data-testid="stMetricValue"] { font-size: 28px; color: #2c3e50 !important; font-weight: 800; }
    div[data-testid="stMetricLabel"] { color: #555 !important; font-weight: bold; }
    
    .stAlert { background-color: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; }
    
    .stFileUploader {
        background-color: rgba(255,255,255, 0.6);
        border: 2px dashed #2c3e50;
        border-radius: 10px;
        padding: 10px;
    }
    section[data-testid="stSidebar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)


CLASS_TRANSLATIONS = {
    "multi fissure": "Agir Kirik",
    "Multi Fissure": "Agir Kirik",
    "multifissure": "Agir Kirik",
    
    "fissure": "Catlak",
    "Fissure": "Catlak",
    
    "impact": "Darbe",
    "Impact": "Darbe",
    "impact/ezilme": "Darbe",
}


REPORT_TRANSLATIONS = {
    "Agir Kirik": "Ağır Kırık (Çoklu)",
    "Catlak": "Çatlak / Kılcal",
    "Darbe": "Darbe İzi / Ezilme"
}


PRICE_POLICY = {
    "Agir Kirik": 6000,
    "Catlak": 4500,
    "Darbe": 1500
}


@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        return None

def process_and_price(image, model, conf_threshold):
    img_np = np.array(image)
    
    
    results = model(img_np, conf=conf_threshold)[0]
    
    
    res_plotted = results.plot()
    
    
    detected_classes_tr = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        raw_name = results.names[cls_id] 
        
        
        
        tr_name = CLASS_TRANSLATIONS.get(raw_name, CLASS_TRANSLATIONS.get(raw_name.lower(), raw_name))
        
        detected_classes_tr.append(tr_name)
        
    return res_plotted, detected_classes_tr



with st.container():
    c1, c2 = st.columns([1, 6])
    with c1:
        
        st.image("https://ffo3gv1cf3ir.merlincdn.net/SiteAssets/Hakkimizda/genel-bakis/logolarimiz/TURKCELL_DIKEY_ERKEK_LOGO.png?20260122_03", width=90)
    with c2:
        st.title("Pasaj AI | Cihaz Değerleme Asistanı")
        st.markdown("**Yarının Teknoloji Liderleri** - Computer Vision Modülü")

st.divider()


model = load_model(MODEL_DOSYASI)
if model is None:
    st.error(f"⚠️ Model dosyası ({MODEL_DOSYASI}) bulunamadı! Lütfen dosyanın proje klasöründe olduğundan emin olun.")
    st.stop()


uploaded_file = st.file_uploader("Değerlenecek cihazın fotoğrafını yükleyin...", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    col_orig, col_res = st.columns(2)
    
    with col_orig:
        st.markdown("### 📸 Orijinal Fotoğraf")
        st.image(image, use_container_width=True)
        st.write("") 
        analyze_btn = st.button("🔍 Hasar Tespiti ve Fiyatla")
    
    if analyze_btn:
        with col_res:
            with st.spinner("AI ekranı inceliyor, hasar tespiti yapılıyor..."):
                
                res_img, detections_tr = process_and_price(image, model, SABIT_CONFIDENCE)
                
                st.markdown("### AI Tespiti")
                st.image(res_img, use_container_width=True)
        
        st.divider()
        st.markdown("## 📊 Ekspertiz Raporu")
        
        report_container = st.container()
        with report_container:
            if not detections_tr:
                st.success("✅ **MÜKEMMEL:** Cihazda kozmetik bir kusur bulunamadı.")
                st.balloons()
            else:
                
                counts = {i: detections_tr.count(i) for i in detections_tr}
                
                base_price = 30000
                total_deduction = 0
                
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.markdown("#### 🛠️ Hasar Detayı")
                    for tr_name, count in counts.items():
                        
                        
                        display_name = REPORT_TRANSLATIONS.get(tr_name, tr_name)
                        
                        
                        unit_price = PRICE_POLICY.get(tr_name, 0)
                        
                        deduction = count * unit_price
                        total_deduction += deduction
                        
                        st.warning(f"⚠️ {count} x **{display_name}**")
                        
                        if unit_price > 0:
                            st.caption(f"Kesinti: -{deduction:,} TL")
                        else:
                            st.error(f"Fiyat bilgisi eksik: {tr_name}")

                with m2:
                    st.markdown("#### 📉 Durum Analizi")
                    st.metric("Toplam Kesinti", f"-{total_deduction:,} TL")
                    
                    
                    det_keys = list(counts.keys())
                    if "Agir Kirik" in det_keys:
                        st.error("🚨 Durum: **AĞIR HASARLI**")
                    elif "Catlak" in det_keys:
                        st.warning("⚠️ Durum: **ORTA HASARLI**")
                    else:
                        st.info("ℹ️ Durum: **HAFİF KUSURLU**")

                with m3:
                    final_price = max(0, base_price - total_deduction)
                    st.markdown("#### 💰 Pasaj Teklifi")
                    st.metric("Son Fiyat", f"{final_price:,} TL", delta=f"-{total_deduction} TL", delta_color="inverse")

elif not uploaded_file:
    st.info("👆 Başlamak için lütfen yukarıdan bir fotoğraf yükleyin.")