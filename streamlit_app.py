import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import cv2
import shap
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
from dotenv import load_dotenv
from breastcancerimages import extract_features
from tensorflow import keras
import tempfile

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Breast Cancer AI Assistant",
    page_icon="🎗️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize OpenAI/Groq client
try:
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
except Exception as e:
    st.error(f"Error initializing API client: {e}")
    client = None

# ---------------------------
# Load Models & Resources
# ---------------------------
@st.cache_resource
def load_resources():
    try:
        scaler = joblib.load("scaler.pkl")
        model = keras.models.load_model("best_model_ann.h5")
        xgb_model = joblib.load("xgb_model.pkl")
        ohe_encoder = joblib.load("ohe_encoder.pkl")
        
        # Initialize SHAP Explainers
        explainer = shap.TreeExplainer(xgb_model)
        
        background_ann = np.zeros((1, scaler.n_features_in_))
        ann_explainer = shap.KernelExplainer(model.predict, background_ann)
        
        return model, scaler, xgb_model, ohe_encoder, explainer, ann_explainer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

model, scaler, xgb_model, ohe_encoder, explainer, ann_explainer = load_resources()

@st.cache_resource
def load_rag_system():
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
        return vector_db
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        return None

vector_db = load_rag_system()

def get_relevant_context(query: str):
    if not vector_db:
        return ""
    try:
        results = vector_db.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return ""

# ---------------------------
# Navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chatbot", "Image Analysis", "Risk Assessment"])

st.sidebar.markdown("---")
st.sidebar.info("Breast Cancer AI Assistant v1.0")

# ---------------------------
# Chatbot Page
# ---------------------------
if page == "Chatbot":
    st.title("💬 Medical AI Assistant")
    st.markdown("Ask questions about breast cancer, symptoms, treatments, or guidelines.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # RAG Context
            context = get_relevant_context(prompt)
            
            full_prompt = f"""You are a helpful medical AI assistant specializing in breast cancer.
            Use the following context to answer the user's question if relevant:
            
            CONTEXT:
            {context}
            
            USER QUESTION:
            {prompt}
            
            Answer clearly and professionally."""

            try:
                response = client.chat.completions.create(
                    model="llama3-8b-8192", # Using a Groq model
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant."},
                        {"role": "user", "content": full_prompt}
                    ]
                )
                full_response = response.choices[0].message.content
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error generating response: {e}")

# ---------------------------
# Image Analysis Page
# ---------------------------
elif page == "Image Analysis":
    st.title("🔬 Histopathology Image Analysis")
    st.markdown("Upload a breast tissue image for analysis.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                try:
                    # Feature extraction
                    feats = extract_features(img)
                    df = pd.DataFrame([feats])
                    df_features = df.select_dtypes(include=np.number)
                    
                    # Scaling
                    X_scaled = scaler.transform(df_features)
                    
                    # Prediction
                    prob_array = model.predict(X_scaled)
                    prob = prob_array[0][0]
                    label = "Malignant" if prob >= 0.5 else "Benign"
                    color = "red" if prob >= 0.5 else "green"

                    st.markdown(f"### Result: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                    st.progress(float(prob))
                    st.write(f"Probability of Malignancy: {prob:.2%}")

                    # SHAP Explanation
                    st.subheader("Explanation (XAI)")
                    if ann_explainer:
                        shap_values = ann_explainer.shap_values(X_scaled, nsamples=100)
                        vals = shap_values[0] if isinstance(shap_values, list) else shap_values
                        if len(vals.shape) > 1:
                            vals = vals[0]
                        
                        feature_names = df_features.columns.tolist()
                        contributions = list(zip(feature_names, vals))
                        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                        
                        top_factors = contributions[:5]
                        for feat, val in top_factors:
                            direction = "increases risk" if val > 0 else "supports benign"
                            st.write(f"- **{feat}**: {direction} (impact: {val:.2f})")

                    # AI Interpretation
                    st.subheader("AI Interpretation")
                    context = get_relevant_context(f"Breast cancer diagnosis {label}")
                    
                    ai_response = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "system", "content": "You are a medical AI. Interpret the diagnostic results."},
                            {"role": "user", "content": f"Diagnosis: {label} ({prob:.2%}). Context: {context}"}
                        ]
                    )
                    st.markdown(ai_response.choices[0].message.content)

                except Exception as e:
                    st.error(f"Error during analysis: {e}")

# ---------------------------
# Risk Assessment Page
# ---------------------------
elif page == "Risk Assessment":
    st.title("📊 Risk Assessment Calculator")
    st.markdown("Enter patient data to estimate breast cancer risk.")

    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            agegrp = st.number_input("Age Group (1-10)", min_value=1, max_value=10, value=4)
            menopaus = st.selectbox("Menopause Status", [0, 1], format_func=lambda x: "Pre-menopausal" if x==0 else "Post-menopausal")
            density = st.selectbox("Breast Density (1-4)", [1, 2, 3, 4])
            race = st.selectbox("Race", [1, 2, 3, 4, 5]) # Need mapping if available
            Hispanic = st.selectbox("Hispanic", [0, 1])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

        with col2:
            agefirst = st.number_input("Age at First Birth", min_value=0, max_value=50, value=25)
            nrelbc = st.selectbox("Relatives with Breast Cancer", [0, 1, 2])
            brstproc = st.selectbox("Previous Breast Procedures", [0, 1])
            lastmamm = st.selectbox("Last Mammogram Result", [0, 1])
            surgmeno = st.selectbox("Surgical Menopause", [0, 1])
            hrt = st.selectbox("Hormone Replacement Therapy", [0, 1])

        submitted = st.form_submit_button("Assess Risk")

    if submitted:
        with st.spinner("Calculating risk..."):
            try:
                # Prepare data
                input_data = {
                    "menopaus": menopaus, "agegrp": agegrp, "density": density,
                    "race": race, "Hispanic": Hispanic, "bmi": bmi,
                    "agefirst": agefirst, "nrelbc": nrelbc, "brstproc": brstproc,
                    "lastmamm": lastmamm, "surgmeno": surgmeno, "hrt": hrt
                }
                
                feature_cols = ["menopaus", "agegrp", "density", "race", "Hispanic", "bmi", "agefirst", "nrelbc", "brstproc", "lastmamm", "surgmeno", "hrt"]
                df = pd.DataFrame([input_data])
                df = df[feature_cols]
                
                # Encode
                X_encoded = ohe_encoder.transform(df)
                
                # Predict
                probability = xgb_model.predict_proba(X_encoded)[0][1]
                
                st.metric("Risk Probability", f"{probability:.2%}")
                
                if probability > 0.5:
                    st.error("High Risk Detected")
                else:
                    st.success("Low Risk Detected")

                # SHAP
                st.subheader("Key Risk Factors")
                if explainer:
                    shap_values = explainer.shap_values(X_encoded)
                    feature_names = ohe_encoder.get_feature_names_out(feature_cols)
                    contributions = list(zip(feature_names, shap_values[0]))
                    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    for feat, val in contributions[:5]:
                        direction = "increases risk" if val > 0 else "decreases risk"
                        st.write(f"- **{feat}**: {direction} (impact: {val:.2f})")

                # AI Advice
                st.subheader("AI Recommendations")
                context = get_relevant_context("breast cancer risk factors")
                ai_response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": "You are a medical AI. Provide recommendations based on risk assessment."},
                        {"role": "user", "content": f"Risk: {probability:.2%}. Patient Data: {input_data}. Context: {context}"}
                    ]
                )
                st.markdown(ai_response.choices[0].message.content)

            except Exception as e:
                st.error(f"Error in calculation: {e}")
