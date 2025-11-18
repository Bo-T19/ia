import os
import chromadb
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from llama_index.core import Settings, VectorStoreIndex,PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()

# --- 1. CONFIGURACIÓN ---
# Carga las claves de API 
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# Configura el LLM y el Embedding Model
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# --- 2. CONFIGURACIÓN DE SEGURIDAD (API TOKEN) ---

# Carga el token de API desde el archivo .env
API_TOKEN_CORRECTO = os.environ.get("API_TOKEN")

# Define el esquema de seguridad: esperamos un header llamado "X-API-Token"
api_key_header = APIKeyHeader(name="X-API-Token")

# Función de dependencia que valida el token
async def verificar_api_token(api_key: str = Security(api_key_header)):
    if not API_TOKEN_CORRECTO:
        # Error de seguridad si el token no está configurado en el servidor
        raise HTTPException(
            status_code=500, 
            detail="Error del servidor: API Token no configurado."
        )
    if api_key != API_TOKEN_CORRECTO:
        # Error si el cliente envía un token incorrecto
        raise HTTPException(
            status_code=401, 
            detail="Token de API inválido o faltante"
        )
    return api_key

# --- 3. INICIALIZACIÓN DE LA APP FASTAPI ---
app = FastAPI(
    title="API de Consulta del POT de Bogotá",
    description="Una API para hacer preguntas en lenguaje natural sobre los documentos del POT.",
    version="1.0"
)

# Variable global para el motor de consulta
query_engine = None

# --- 4. CARGA DEL ÍNDICE AL INICIAR LA API ---
@app.on_event("startup")
async def startup_event():
    global query_engine
    
    print("Iniciando API... Cargando el índice vectorial desde './chroma_db'...")

    # Prompt para la respuesta inicial (Text QA)
    qa_prompt_str = (
        "La siguiente es información de contexto del POT de Bogotá.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Dada la información de contexto y SIN usar conocimiento previo, "
        "responde la siguiente pregunta.\n"
        "Si la respuesta no está en el contexto, di: 'Lo siento, no he encontrado información al respecto en los documentos del POT. Trata de escribir la pregunta en otros términos para hacer una búsqueda más efectiva en los documentos del POT'\n"
        "Pregunta: {query_str}\n"
        "Respuesta:"
    )

    # Prompt para refinar la respuesta (si encuentra más fragmentos relevantes)
    refine_prompt_str = (
        "La pregunta original es: {query_str}\n"
        "Tenemos una respuesta existente: {existing_answer}\n"
        "Tenemos la oportunidad de refinar la respuesta existente "
        "(solo si es necesario) con más contexto a continuación.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Dado el nuevo contexto, refina la respuesta original para responder mejor la pregunta. "
        "Si el contexto no es útil, devuelve la respuesta original. "
        "No uses conocimiento previo fuera de este contexto.\n"
        "Respuesta Refinada:"
    )

    # Convertirlos a objetos PromptTemplate
    text_qa_template = PromptTemplate(qa_prompt_str)
    refine_template = PromptTemplate(refine_prompt_str)
        
    # Conecta a la base de datos ChromaDB local
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_collection("pot_bogota")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Carga el índice desde el vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model
    )
    
    # --- LA LÓGICA CLAVE DE RAG ---
    
    # 1. Crear el Retriever (Recuperador)
    retriever = index.as_retriever(
        similarity_top_k=5,
        retriever_mode="recursive"
    )
    
    # 2. Crear el Motor de Consulta (Query Engine)
    query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=Settings.llm,
            text_qa_template=text_qa_template, 
            refine_template=refine_template, 
            verbose=True
        )
    
    print("¡Índice cargado y motor de consulta listo!")

# --- 5. MODELOS DE DATOS (REQUEST/RESPONSE) ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# --- 6. EL ENDPOINT DE LA API (PROTEGIDO) ---
@app.post("/query", 
          response_model=QueryResponse,
          dependencies=[Depends(verificar_api_token)] 
         )
async def query_pot(request: QueryRequest):
    """
    Recibe una pregunta en lenguaje natural y consulta el RAG del POT.
    (Solo con token de API válido)
    """
    if not query_engine:
        # Esto no debería pasar si el startup_event funcionó
        return QueryResponse(answer="Error: El motor de consulta no está inicializado.")
        
    print(f"Recibida consulta (autenticada): {request.question}")
    
    # Ejecuta la consulta
    response = await query_engine.aquery(request.question)
    
    return QueryResponse(answer=str(response))

# --- 7. ENDPOINT DE BIENVENIDA (OPCIONAL) ---
@app.get("/")
def read_root():
    return {"message": "API del POT de Bogotá está activa. Use el endpoint /query para hacer preguntas."}