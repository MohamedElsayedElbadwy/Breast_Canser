import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from tensorflow.keras.models import load_model

# --- إعدادات الصفحة الأساسية ---
st.set_page_config(
    page_title="BCW Prediction System",
    page_icon="🎗️",
    layout="wide"
)

# --- تنسيق واجهة المستخدم (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #d33682; color: white; font-weight: bold; }
    .prediction-box { padding: 20px; border-radius: 15px; text-align: center; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- تحميل الموديل والسكيلر (مرة واحدة) ---
@st.cache_resource
def load_my_assets():
    # تحميل الموديل والسكيلر اللي أنت عملتهم في النوت بوك
    model = load_model('breast_cancer_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    nn_model, my_scaler = load_my_assets()
    st.sidebar.success("✅ Model & Scaler Loaded")
except Exception as e:
    st.sidebar.error("❌ Files Missing: Make sure 'model.h5' and 'scaler.pkl' are in the folder.")
    st.stop()

# --- القائمة الجانبية ---
st.sidebar.title("Navigation")
menu = ["Home", "Prediction Tool", "About"]
choice = st.sidebar.selectbox("Go to", menu)

# --- الصفحة الرئيسية ---
if choice == "Home":
    st.title("🎗️ Breast Cancer Wisconsin Prediction System")
    st.markdown("""
    ### Welcome, Engineer Mohamed Elsayed
    هذا النظام يستخدم تقنيات التعلم العميق (Deep Learning) للتنبؤ بنوع الورم بناءً على بيانات Wisconsin Breast Cancer.
    
    **كيف يعمل النظام؟**
    1. يتم إدخال الخصائص الحيوية للورم.
    2. يقوم النظام بعمل Scaling للبيانات بنفس معايير التدريب.
    3. يقوم موديل Keras بتحليل البيانات وإعطاء النتيجة النهائية.
    """)
    st.image("https://www.verywellhealth.com/thmb/m_3_vU9_6X0H8_B-Wf6-W9z_9k0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/breast-cancer-biopsy-results-5a8f2e8f1d4be80036f0a9a4.jpg")

# --- صفحة التوقع ---
elif choice == "Prediction Tool":
    st.title("🧠 Diagnostic Tool (Keras Engine)")
    st.write("Enter the following features (as calculated in your notebook):")

    ## --- 1. واجهة المدخلات (الـ 19 ميزة اللي في الصورة بالظبط) ---
col1, col2, col3 = st.columns(3)

with col1:
    f1 = st.number_input("Radius Mean", value=14.0)
    f2 = st.number_input("Texture Mean", value=19.0)
    f3 = st.number_input("Smoothness Mean", value=0.1)
    f4 = st.number_input("Compactness Mean", value=0.1)
    f5 = st.number_input("Concavity Mean", value=0.1)
    f6 = st.number_input("Concave Points Mean", value=0.05)
    f7 = st.number_input("Symmetry Mean", value=0.1)

with col2:
    f8 = st.number_input("Radius SE", value=0.4)
    f9 = st.number_input("Compactness SE", value=0.02)
    f10 = st.number_input("Concavity SE", value=0.03)
    f11 = st.number_input("Concave Points SE", value=0.01)
    f12 = st.number_input("Radius Worst", value=16.0)
    f13 = st.number_input("Texture Worst", value=25.0)

with col3:
    f14 = st.number_input("Smoothness Worst", value=0.1)
    f15 = st.number_input("Compactness Worst", value=0.2)
    f16 = st.number_input("Concavity Worst", value=0.2)
    f17 = st.number_input("Concave Points Worst", value=0.1)
    f18 = st.number_input("Symmetry Worst", value=0.2)
    f19 = st.number_input("Fractal Dimension Worst", value=0.08)

# --- 2. التجميع بالترتيب القاتل (نفس ترتيب الصورة) ---
# لازم الترتيب ده يكون مراية للـ Notebook عشان الـ Scaler ميتلخبطش
user_features = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19]



if st.button("Run AI Diagnosis"):
        # 1. تحويل لـ Numpy وتغيير الشكل ليتناسب مع الموديل
        input_array = np.array([user_features])
        
        # 2. عملية الـ Scaling (السر في نجاح التوقع)
        try:
            # لو الموديل متدرب على 30 ميزة وأنت مدخل 6 بس، السكيلر هيطلع Error
            # تأكد إن عدد المدخلات في user_features هو نفس عدد أعمدة الـ X_train
            input_scaled = my_scaler.transform(input_array)
            
            # 3. التوقع
            prediction_prob = nn_model.predict(input_scaled)
            is_malignant = prediction_prob[0][0] >= 0.5

            st.markdown("---")
            if is_malignant:
                st.error("### Result: Malignant (خبيث)")
                st.progress(float(prediction_prob[0][0]))
            else:
                st.success("### Result: Benign (حميد)")
                st.progress(float(prediction_prob[0][0]))
            
            st.write(f"**AI Confidence Score:** {float(prediction_prob[0][0])*100:.2f}%")
            
        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.warning("تأكد أن عدد المدخلات يطابق ما تدرب عليه الـ Scaler والموديل (مثلاً 30 ميزة).")

# --- صفحة المعلومات ---
elif choice == "About":
    st.subheader("Project Details")
    st.info(f"Developed by: Mohamed Elsayed Elbadwy Mansour")
    st.write("Student Code: 324250065")
    st.write("This project utilizes a Neural Network built with Keras API and deployed via Streamlit.")