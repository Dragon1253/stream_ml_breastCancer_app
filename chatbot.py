from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
from openai import OpenAI
import os
import joblib
import numpy as np
import pandas as pd
import cv2
import shap
# IMPORTS RAG & AI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import json
import os
from datetime import datetime
#elyes-----------------------------------------------------------

import uvicorn
import cv2
import numpy as np
import joblib
import pandas as pd
# Assurez-vous que le fichier breastcancerimages.py est au même niveau que api.py
from breastcancerimages import extract_features 
from tensorflow import keras 
import joblib 
import shap # <--- IMPORT SHAP
# -----------------------------------------------------

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# OPTIMISATION : Ne jamais laisser la clé API dans le code source.
# Utilisez os.getenv("OPENAI_API_KEY") ou un fichier .env
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY") 
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/patient_images", StaticFiles(directory="patient_images"), name="patient_images")

# ---------------------------
# Chargement des modèles et du scaler (Effectué une seule fois au démarrage)
# ---------------------------
try:
    # 1. CHARGEMENT DU SCALER (CRUCIAL POUR L'ANN)
    scaler = joblib.load("scaler.pkl") 
    print("✔️ Scaler chargé.")

    # 2. CHARGEMENT DU MODÈLE ANN (Keras)
    # Note: Keras charge le modèle H5 correctement, résolvant le NameError précédent.
    model = keras.models.load_model("best_model_ann.h5")
    print("✔️ Modèle ANN chargé.")

    # 3. CHARGEMENT DU MODÈLE XGBOOST
    xgb_model = joblib.load("xgb_model.pkl")
    print("✔️ Modèle XGBoost chargé.")

    # 4. CHARGEMENT DE L'ENCODEUR OHE
    ohe_encoder = joblib.load("ohe_encoder.pkl")
    print("✔️ Encodeur OHE chargé.")

    # 5. INITIALISATION DE L'EXPLAINER SHAP (NOUVEAU)
    # TreeExplainer est optimisé pour XGBoost
    print("⏳ Initialisation de SHAP Explainer (XGBoost)...")
    explainer = shap.TreeExplainer(xgb_model)
    print("✔️ SHAP Explainer (XGBoost) prêt.")

    # 6. INITIALISATION DE L'EXPLAINER SHAP POUR ANN (NOUVEAU)
    # On utilise KernelExplainer pour le modèle Keras (Black-box approach compatible)
    # On utilise un background de zéros car les données sont standardisées (mean ~ 0)
    print("⏳ Initialisation de SHAP Explainer (ANN)...")
    # scaler.n_features_in_ nous donne le nombre de features attendues (26 normalement)
    background_ann = np.zeros((1, scaler.n_features_in_))
    ann_explainer = shap.KernelExplainer(model.predict, background_ann)
    print("✔️ SHAP Explainer (ANN) prêt.")

except Exception as e:
    print(f"❌ ERREUR DE CHARGEMENT : Le modèle ou le scaler n'a pas pu être chargé. Assurez-vous que 'best_model_ann.h5', 'scaler.pkl', 'xgb_model.pkl' et 'ohe_encoder.pkl' existent. Détails: {e}")
    # Définir à None pour gérer les erreurs dans les routes
    model = None
    scaler = None
    xgb_model = None
    ohe_encoder = None
    explainer = None 
    ann_explainer = None

# -----------------------------------------------------
# 2. RAG SETUP (Mémoire des documents)
# -----------------------------------------------------
# Doit être identique à ingest_data.py
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

def get_relevant_context(query: str):
    """Cherche les 3 passages les plus pertinents dans la base vectorielle."""
    try:
        results = vector_db.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        print(f"Erreur RAG: {e}")
        return ""
# -----------------------------------------------------
# IMAGE ANALYSIS — statique pour maintenant
# -----------------------------------------------------
def analyze_image(image_bytes: bytes):
    
   # Vérification que le modèle est chargé
    if model is None or scaler is None:
         raise HTTPException(status_code=500, detail="Modèle ou Scaler non chargé. Veuillez vérifier les logs au démarrage.")

    # Lire l'image
    image_data = image_bytes
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Fichier image non valide.")

    # 1. Extraire les features
    feats = extract_features(img)
    df = pd.DataFrame([feats])
    
    # 2. STANDARDISER LES FEATURES (CRUCIAL POUR L'ANN)
    # Supprimer les colonnes non-numériques si elles existent (comme 'patient' ou 'filename')
    df_features = df.select_dtypes(include=np.number) 
    
    # Standardisation
    X_scaled = scaler.transform(df_features)

    # 3. Prédiction (Utiliser model.predict() pour Keras)
    # model.predict() retourne un tableau numpy, nous voulons le premier élément [0]
    prob_array = model.predict(X_scaled) 
    
    # La probabilité est la première (et unique) valeur du résultat [0][0]
    prob = prob_array[0][0] 
    
    # Déterminer la classe (0 ou 1)
    label = int(prob >= 0.5)

    # --- XAI ANN : CALCUL SHAP ---
    explanation_text = "No explanation available."
    if ann_explainer:
        try:
            # Calcul des valeurs SHAP
            # nsamples='auto' ou un petit nombre pour la rapidité avec KernelExplainer
            shap_values = ann_explainer.shap_values(X_scaled, nsamples=100)
            
            # Gestion du format de retour de shap_values (liste ou array)
            # Pour un modèle binaire Keras, c'est souvent une liste [array_class_0] ou [array_output]
            vals = shap_values[0] if isinstance(shap_values, list) else shap_values
            
            # Si vals est (1, n_features), on l'aplatit
            if len(vals.shape) > 1:
                vals = vals[0]
            
            # Récupérer les noms des features
            feature_names = df_features.columns.tolist()
            
            # Associer et trier
            contributions = list(zip(feature_names, vals))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Top 5
            top_factors = contributions[:5]
            
            factors_desc = []
            for feat, val in top_factors:
                # Interprétation : val > 0 pousse vers 1 (Maligne), val < 0 pousse vers 0 (Bénigne)
                direction = "increases malignancy risk" if val > 0 else "supports benign diagnosis"
                factors_desc.append(f"- {feat}: {direction} (impact: {val:.2f})")
            
            explanation_text = "\n".join(factors_desc)
        except Exception as e:
            print(f"XAI ANN Error: {e}")

    # 4. Retourner le résultat
    return {
        "prediction": label,
        "probability": float(prob),
        "explanation": explanation_text,
        # Retourner les features brutes pour affichage de la table HTML
        "features": {k: float(v) for k, v in feats.items() if k not in ['patient', 'filename']} 
    }

# -----------------------------------------------------
# ROUTE: Upload image
# -----------------------------------------------------
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    raw_diagnostic = analyze_image(raw_bytes)

    # RAG: Récupérer le contexte basé sur le diagnostic
    context = get_relevant_context(raw_diagnostic)
    if not context:
        context = "No specific guidelines found."

    # Un seul appel pour reformuler avec le contexte
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b", # Utilisation d'un modèle rapide et standard
        messages=[
            {"role": "system", "content": f"""You are a medical AI. 
            1. Analyze the image diagnostic (Prediction: {raw_diagnostic['prediction']}, Probability: {raw_diagnostic['probability']:.2%}).
            2. **Explain WHY (XAI)**: Use the provided 'XAI Factors' to explain which image features (texture, geometry, etc.) influenced the model. Translate technical terms (like 'ASM', 'contrast') into simple medical explanations.
            3. Provide recommendations based on this CONTEXT:\n{context}. 
            
            IMPORTANT: Format your response using Markdown. 
            - Use '### Diagnostic Summary' for the main result.
            - Use '### Key Image Features (XAI)' for the explanation.
            - Use '### Recommendations' for next steps.
            - Use bullet points."""},
            {"role": "user", "content": f"Diagnostic Data: {raw_diagnostic}. XAI Factors: \n{raw_diagnostic['explanation']}"}
        ]
    )
    reformulated = response.choices[0].message.content

    return JSONResponse({
        "status": "ok",
        "diagnostic_raw": raw_diagnostic,
        "diagnostic_ai": reformulated
    })

# -------------------------------------------
# Risk estimator (AVEC XAI)
# -------------------------------------------
def estimate_risk(vars):
    if xgb_model is None or ohe_encoder is None:
        return {"error": "Model or Encoder not loaded", "probability": 0.95, "explanation": "N/A"}

    try:
        # Liste des features dans l'ordre attendu par le modèle
        feature_cols = ["menopaus", "agegrp", "density", "race", "Hispanic", "bmi", "agefirst", "nrelbc", "brstproc", "lastmamm", "surgmeno", "hrt"]
        
        # Création du DataFrame
        df = pd.DataFrame([vars])
        
        # Sélection et ordonnancement des colonnes
        df = df[feature_cols]
        
        # Conversion en numérique (au cas où)
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Application de l'encodage One-Hot
        X_encoded = ohe_encoder.transform(df)
        
        # Prédiction
        prediction = xgb_model.predict(X_encoded)[0]
        probability = xgb_model.predict_proba(X_encoded)[0][1]
        
        # --- XAI : CALCUL SHAP ---
        explanation_text = "No explanation available."
        if explainer:
            # Calcul des valeurs SHAP pour cette instance spécifique
            shap_values = explainer.shap_values(X_encoded)
            
            # Récupérer les noms des features après encodage One-Hot
            feature_names = ohe_encoder.get_feature_names_out(feature_cols)
            
            # Créer une liste de (feature, impact)
            # shap_values[0] car nous avons une seule instance
            contributions = list(zip(feature_names, shap_values[0]))
            
            # Trier par valeur absolue d'impact (les plus importants en premier)
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Prendre les 5 facteurs les plus influents
            top_factors = contributions[:5]
            
            # Construire un texte descriptif pour le LLM
            factors_desc = []
            for feat, val in top_factors:
                direction = "increases risk" if val > 0 else "decreases risk"
                # Nettoyage du nom de la feature pour être plus lisible (ex: "agegrp_2" -> "agegrp_2")
                factors_desc.append(f"- {feat}: {direction} (impact: {val:.2f})")
            
            explanation_text = "\n".join(factors_desc)

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "explanation": explanation_text # On retourne l'explication brute
        }
    except Exception as e:
        print(f"Erreur dans estimate_risk: {e}")
        return {"error": str(e), "probability": 0.95, "explanation": "Error calculating explanation"}

class RiskRequest(BaseModel):
    menopaus: int
    agegrp: int
    density: int
    race: int
    Hispanic: int
    bmi: int
    agefirst: int
    nrelbc: int
    brstproc: int
    lastmamm: int
    surgmeno: int
    hrt: int

@app.post("/assess-risk")
async def assess_risk_endpoint(data: RiskRequest):
    vars_dict = data.dict()
    
    # 1. Estimer le risque
    risk_score_data = estimate_risk(vars_dict)
    
    risk_val = risk_score_data.get("probability", 0)
    explanation_xai = risk_score_data.get("explanation", "")

    # 2. Interprétation LLM
    # RAG Context (Generic or based on high risk factors)
    context = get_relevant_context("breast cancer risk factors assessment")
    
    analysis = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": f"""You are a medical AI. 
            1. Interpret the risk score ({risk_val:.2%}) for the doctor.
            2. **Explain WHY**: Use the provided 'XAI Factors' to explain which patient characteristics drove this prediction.
            3. Provide recommendations based on this CONTEXT:\n{context}
            
            IMPORTANT: Format your response using Markdown. 
            - Use '### Analysis' for the interpretation.
            - Use '### Key Risk Factors (XAI)' for the explanation.
            - Use '### Recommendations' for next steps.
            - Use bullet points."""},
            {"role": "user", "content": f"Patient data: {vars_dict}. XAI Factors: \n{explanation_xai}"}
        ]
    ).choices[0].message.content

    return JSONResponse({
        "risk_score": risk_score_data,
        "analysis": analysis
    })

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    patient_name: str = None # Nouveau champ optionnel

# -------------------------------------------
# Route /chat (OPTIMISÉE)
# -------------------------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Optimisation : 1 seul appel API gère l'intention, l'extraction et la réponse conversationnelle.
    """
    message = request.message
    history = request.history
    patient_name = request.patient_name # Récupérer le nom

    # 1. Sauvegarder le message de l'utilisateur
    if patient_name:
        save_message(patient_name, "user", "text", message)
    
    # RAG: Récupérer le contexte AVANT l'appel LLM
    context = get_relevant_context(message)
    if not context:
        context = "No specific guidelines found."

    system_prompt = (
        "You are a medical AI assistant. Analyze the doctor's message.\n"
        f"CONTEXT FROM MEDICAL GUIDELINES:\n{context}\n\n"
        "You must output a JSON object with the following structure:\n"
        "{\n"
        "  'intent': 'risk_evaluation' or 'normal_chat',\n"
        "  'extracted_variables': {\n"
        "      'menopaus': int (0=pré-ménopause, 1=post-ménopause ou âge≥55, 9=inconnu) or null,\n"
        "      'agegrp': int (1=35-39, 2=40-44, ... , 10=80-84) or null,\n"
        "      'density': int (1=entièrement graisse, 2=diffuse, 3=hétérogène, 4=extrêmement dense, 9=inconnu) or null,\n"
        "      'race': int (1=white, 2=Asian/Pacific Islander, 3=black, 4=Native American, 5=other/mixed, 9=unknown) or null,\n"
        "      'Hispanic': int (0=non, 1=oui, 9=inconnu) or null,\n"
        "      'bmi': int (1=10-24.99, 2=25-29.99, 3=30-34.99, 4=35+, 9=inconnu) or null,\n"
        "      'agefirst': int (0=<30, 1=≥30, 2=nullipare, 9=inconnu) or null,\n"
        "      'nrelbc': int (0=aucun, 1=un, 2=2+, 9=inconnu) or null,\n"
        "      'brstproc': int (0=non, 1=oui, 9=inconnu) or null,\n"
        "      'lastmamm': int (0=négatif, 1=faux positif, 9=inconnu) or null,\n"
        "      'surgmeno': int (0=naturel, 1=surgical, 9=inconnu ou non ménopausée) or null,\n"
        "      'hrt': int (0=non, 1=oui, 9=inconnu ou non ménopausée) or null\n"
        "  },\n"
        "  'response_text': string\n"
        "}\n"
        "RULES:\n"
        "1. If intent is 'normal_chat', put your answer in 'response_text' USING THE CONTEXT provided.\n"
        "2. If intent is 'risk_evaluation', extract variables.\n"
        "3. If variables are missing for risk evaluation, 'response_text' MUST be a polite question asking for them.\n"
        "4. If all variables are present, 'response_text' can be empty or a confirmation."
    )

    # Construction de l'historique pour le contexte
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if msg.get("role") in ["user", "assistant"] and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    # APPEL UNIQUE : Classification + Extraction + Génération de réponse
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b", # Modèle rapide et économique
        response_format={"type": "json_object"},
        messages=messages
    )

    # Parsing du résultat
    try:
        result_json = json.loads(completion.choices[0].message.content)
    except:
        # Fallback si le JSON est mal formé
        return JSONResponse({
            "intent": "normal_chat",
            "extracted_variables": {},
            "missing": [],
            "response": "I apologize, I encountered an error processing your request."
        })
    
    intent = result_json.get("intent")
    vars_extracted = result_json.get("extracted_variables")
    if vars_extracted is None:
        vars_extracted = {}
    response_text = result_json.get("response_text")

    # CAS 1 : Conversation normale ou Variables manquantes
    # Si le LLM a déjà généré une question pour les variables manquantes, on la renvoie directement.
    required_vars = ["menopaus", "agegrp", "density", "race", "Hispanic", "bmi", "agefirst", "nrelbc", "brstproc", "lastmamm", "surgmeno", "hrt"]
    missing = [v for v in required_vars if vars_extracted.get(v) in [None, "", "null"]]

    if intent == "normal_chat" or len(missing) > 0:
        # Fix: Ensure response is not empty
        if not response_text:
            if len(missing) > 0:
                response_text = f"Could you please provide the following details: {', '.join(missing)}?"
            else:
                response_text = "I understand. How else can I assist you?"

        if patient_name:
            save_message(patient_name, "assistant", "text", response_text)

        return JSONResponse({
            "intent": intent,
            "extracted_variables": vars_extracted,
            "missing": missing,
            "response": response_text # Le LLM a déjà formulé la réponse/question ici
        })

    # CAS 2 : Toutes les variables sont là -> Calcul du risque
    # C'est le seul cas où on pourrait avoir besoin d'un 2ème appel pour interpréter le résultat chiffré
    risk_score_data = estimate_risk(vars_extracted) # Récupère le dict complet avec explication
    
    risk_val = risk_score_data.get("probability", 0)
    explanation_xai = risk_score_data.get("explanation", "")

    # Appel final pour l'interprétation médicale du score (inévitable si on veut une analyse textuelle du chiffre)
    final_analysis = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": f"""You are a medical AI. 
            1. Interpret the risk score ({risk_val:.2%}) for the doctor.
            2. **Explain WHY**: Use the provided 'XAI Factors' to explain which patient characteristics drove this prediction. Translate technical feature names (like 'agegrp_1') into natural language.
            3. Provide recommendations based on this CONTEXT:\n{context}
            
            IMPORTANT: Format your response using Markdown. 
            - Use '### Analysis' for the interpretation.
            - Use '### Key Risk Factors (XAI)' for the explanation of why.
            - Use '### Recommendations' for next steps.
            - Use bullet points."""},
            {"role": "user", "content": f"Patient data: {vars_extracted}. XAI Factors (SHAP values): \n{explanation_xai}. Generate summary."}
        ]
    ).choices[0].message.content

    if patient_name:
        # Sauvegarder comme une "risk_card" avec les données structurées
        save_message(patient_name, "assistant", "risk_card", final_analysis, extra_data=risk_score_data)

    return JSONResponse({
        "intent": "risk_evaluation",
        "risk_score": risk_score_data,
        "extracted_variables": vars_extracted,
        "response": final_analysis
    })
# Dossier pour stocker les historiques de chat
CHAT_HISTORY_DIR = "chat_histories"
if not os.path.exists(CHAT_HISTORY_DIR):
    os.makedirs(CHAT_HISTORY_DIR)

def get_chat_filename(patient_name):
    # Nettoyage du nom pour le nom de fichier
    safe_name = "".join([c for c in patient_name if c.isalpha() or c.isdigit() or c==' ']).strip().replace(' ', '_')
    return os.path.join(CHAT_HISTORY_DIR, f"{safe_name}.json")

def save_message(patient_name, role, message_type, content, extra_data=None):
    """
    Sauvegarde un message structuré.
    message_type: 'text', 'risk_card', 'image'
    extra_data: dict pour stocker les scores, probabilités, chemin image, etc.
    """
    if not patient_name:
        return

    filename = get_chat_filename(patient_name)
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role, # 'user' ou 'assistant'
        "type": message_type,
        "content": content,
        "data": extra_data or {}
    }

    history = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                history = json.load(f)
            except:
                history = []
    
    history.append(entry)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

@app.get("/history/{patient_name}")
async def get_history(patient_name: str):
    filename = get_chat_filename(patient_name)
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return JSONResponse(json.load(f))
    return JSONResponse([])

@app.get("/")
async def root():
    return FileResponse('dashboard.html')

@app.get("/chat_ui")
async def chat_ui():
    return FileResponse('index.html')

@app.get("/patients_data.csv")
async def get_csv():
    return FileResponse('patients_data.csv')

@app.get("/dashboard.html")
async def dashboard_file():
    return FileResponse('dashboard.html')

