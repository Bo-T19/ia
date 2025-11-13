import os
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
load_dotenv()

# --- 1. CONFIGURACIÓN ---
# Carga las claves de API (igual que en tu script de ingesta)
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# Configura el LLM y el Embedding Model
# (Referencia: Components_Of_LlamaIndex.ipynb)
Settings.llm = OpenAI(model="gpt-4o-mini") # gpt-4o-mini es rápido y barato
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# --- 2. INICIALIZACIÓN DE LA APP FASTAPI ---
app = FastAPI(
    title="API de Consulta del POT de Bogotá",
    description="Una API para hacer preguntas en lenguaje natural sobre los documentos del POT.",
    version="1.0"
)

# Variable global para el motor de consulta
query_engine = None

# --- 3. CARGA DEL ÍNDICE AL INICIAR LA API ---
@app.on_event("startup")
async def startup_event():
    global query_engine
    
    print("Iniciando API... Cargando el índice vectorial desde './chroma_db'...")
    
    # Conecta a la base de datos ChromaDB local
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_collection("pot_bogota")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Carga el índice desde el vector store
    # (Referencia: Ingestion_Pipeline.ipynb)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model
    )
    
    # --- LA LÓGICA CLAVE DE RAG ---
    
    # 1. Crear el Retriever (Recuperador)
    # Usamos "recursive" para que pueda consultar las TABLAS
    # (Referencia: Advanced_RAG_with_LlamaParse.ipynb)
    retriever = index.as_retriever(
        similarity_top_k=5,
        retriever_mode="recursive" # ¡Clave para las tablas!
    )
    
    # 2. Crear el Motor de Consulta (Query Engine)
    # (Referencia: Components_Of_LlamaIndex.ipynb)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=Settings.llm,
        verbose=True
    )
    
    print("¡Índice cargado y motor de consulta listo!")

# --- 4. MODELOS DE DATOS (REQUEST/RESPONSE) ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# --- 5. EL ENDPOINT DE LA API ---
@app.post("/query", response_model=QueryResponse)
async def query_pot(request: QueryRequest):
    """
    Recibe una pregunta en lenguaje natural y consulta el RAG del POT.
    """
    if not query_engine:
        # Esto no debería pasar si el startup_event funcionó
        return QueryResponse(answer="Error: El motor de consulta no está inicializado.")
        
    print(f"Recibida consulta: {request.question}")
    
    # Ejecuta la consulta
    response = await query_engine.aquery(request.question)
    
    return QueryResponse(answer=str(response))

# --- 6. ENDPOINT DE BIENVENIDA (OPCIONAL) ---
@app.get("/")
def read_root():
    return {"message": "API del POT de Bogotá está activa. Use el endpoint /query para hacer preguntas."}