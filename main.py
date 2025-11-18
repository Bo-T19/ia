import os
import chromadb
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine # <--- IMPORTANTE
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

# --- 1. CONFIGURACIÓN ---
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1) # Subimos un poco la temp para que sea más natural
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# --- 2. CONFIGURACIÓN DE SEGURIDAD (API TOKEN) ---
API_TOKEN_CORRECTO = os.environ.get("API_TOKEN")
api_key_header = APIKeyHeader(name="X-API-Token")

async def verificar_api_token(api_key: str = Security(api_key_header)):
    if not API_TOKEN_CORRECTO:
        raise HTTPException(status_code=500, detail="Error del servidor: API Token no configurado.")
    if api_key != API_TOKEN_CORRECTO:
        raise HTTPException(status_code=401, detail="Token de API inválido o faltante")
    return api_key

# --- 3. INICIALIZACIÓN DE LA APP FASTAPI ---
app = FastAPI(
    title="Chatbot Experto del POT de Bogotá",
    description="Un asistente conversacional sobre los documentos del POT.",
    version="2.0" # Cambiamos a ChatEngine
)

# Variable global para el motor de CHAT
chat_engine = None # Ya no es 'query_engine'

# --- 4. CARGA DEL ÍNDICE AL INICIAR LA API ---
@app.on_event("startup")
async def startup_event():
    global chat_engine
    
    print("Iniciando API... Cargando el índice vectorial desde './chroma_db'...")

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_collection("pot_bogota")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model
    )
    
    # --- LA LÓGICA CLAVE DE RAG (CHAT ENGINE) ---
    
    # 1. Crear el Retriever (Sigue siendo recursivo para las tablas)
    retriever = index.as_retriever(
        similarity_top_k=5,
        retriever_mode="recursive" 
    )
    
    # 2. Definir la "Personalidad" del Chatbot (El Template "Suave")
    system_prompt = (
        "Eres un asistente experto y amable especializado en el Plan de Ordenamiento Territorial (POT) de Bogotá. "
        "Tu objetivo es ayudar a los ciudadanos a entender las normas urbanísticas. "
        "Usa ÚNICAMENTE el contexto proporcionado ({context_str}) para responder preguntas técnicas sobre el POT. "
        "Si la pregunta no tiene NADA que ver con el POT o el urbanismo (ej. literatura, ciencia, programación), "
        "responde amablemente: 'Lo siento, solo puedo responder preguntas sobre el POT de Bogotá.' "
        "Si te saludan o te dan las gracias, responde con naturalidad."
    )

    # 3. Configurar la Memoria (Para que recuerde la conversación)
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    # 4. Ensamblar el Chat Engine (en lugar del Query Engine)
    chat_engine = ContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        system_prompt=system_prompt,
        verbose=True
    )
    
    print("¡Índice cargado con Chat Engine + Retriever Recursivo!")

# --- 5. MODELOS DE DATOS (REQUEST/RESPONSE) ---
class QueryRequest(BaseModel):
    question: str

class SourceNode(BaseModel):
    text: str
    file_path: Optional[str] = None
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceNode]


# --- 6. EL ENDPOINT DE LA API (PROTEGIDO) ---
@app.post("/query", 
          response_model=QueryResponse,
          dependencies=[Depends(verificar_api_token)] 
         )
async def query_pot(request: QueryRequest):
    if not chat_engine:
        raise HTTPException(status_code=503, detail="El motor de consulta no está inicializado.")
        
    print(f"Recibida consulta (autenticada): {request.question}")
    
    # --- ¡CAMBIO AQUÍ: Usamos .achat() en lugar de .aquery()! ---
    response = await chat_engine.achat(request.question)
    
    # 1. Extraer la respuesta de texto
    answer = str(response)

    # 2. Extraer los nodos de contexto (fuentes)
    source_nodes_data = []
    if response.source_nodes:
        for node_with_score in response.source_nodes:
            node = node_with_score.node
            source_nodes_data.append(
                SourceNode(
                    text=node.get_content(),
                    file_path=node.metadata.get("file_path", "N/A"), # Obtenemos la ruta del archivo
                    score=node_with_score.score
                )
            )
    
    # 3. Devolver el objeto JSON completo
    return QueryResponse(answer=answer, sources=source_nodes_data)

# --- 7. ENDPOINT DE BIENVENIDA (OPCIONAL) ---
@app.get("/")
def read_root():
    return {"message": "API del POT de Bogotá está activa. Use el endpoint /query para hacer preguntas."}