import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# REMPLACEMENT : On utilise HuggingFace au lieu d'OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 0. Nettoyage (Optionnel mais recommandé)
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("Ancienne base de données supprimée.")

# 1. Charger les PDFs
pdf_files = ["refces_k_du_sein_vf.pdf", "breast-cancer-screening-final-recommendation.pdf"]
documents = []

for pdf_path in pdf_files:
    if os.path.exists(pdf_path):
        print(f"Chargement de {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    else:
        print(f"Attention: Le fichier {pdf_path} n'existe pas.")

if not documents:
    print("Aucun document chargé.")
    exit()

# 2. Découper le texte
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Document découpé en {len(chunks)} morceaux.")

# 3. Créer la base de données vectorielle (GRATUIT & LOCAL)
print("Création des embeddings (cela tourne sur votre CPU, patientez un peu)...")

# Ce modèle est téléchargé une fois et tourne localement. Pas de clé API nécessaire.
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_function, 
    persist_directory="./chroma_db"
)

print("Base de données vectorielle créée avec succès !")