# Configuración de API Keys y entorno
import nest_asyncio
nest_asyncio.apply()

import asyncio
import glob
import os
import time
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.readers.llama_parse import LlamaParse
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from dotenv import load_dotenv
load_dotenv()
import chromadb

# --- 1. CONFIGURACIÓN ---

# Claves de API
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
os.environ["LLAMA_CLOUD_API_KEY"] = os.environ.get("LLAMA_CLOUD_API_KEY")

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

node_parser = MarkdownElementNodeParser(
    llm=Settings.llm, 
    num_workers=1  
)

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("pot_bogota")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Pipeline SÓLO con transformaciones y almacén
pipeline = IngestionPipeline(
    transformations=[node_parser, Settings.embed_model],
    vector_store=vector_store
)

# Lector LlamaParse (lo usaremos archivo por archivo)
reader = LlamaParse(result_type="markdown", verbose=True)

# --- 2. EJECUCIÓN (Bucle archivo por archivo con logging) ---

# Obtenemos todas las rutas de los archivos PDF
all_pdf_files = []
carpetas = [
    "./archivos_base/Bogota/Documentos",
    "./archivos_base/Bogota/Reglamentación",
    "./archivos_base/Bogota/Documentos Técnicos de Soporte",
    "./archivos_base/Bogota/Diagnóstico"
]

for carpeta in carpetas:
    ruta_completa = os.path.join(carpeta, "*.pdf")
    archivos_encontrados = glob.glob(ruta_completa)
    all_pdf_files.extend(archivos_encontrados)
    print(f"Encontrados {len(archivos_encontrados)} archivos en: {carpeta}")

print(f"\nSe encontraron {len(all_pdf_files)} documentos PDF en total para procesar.")

# Función async para procesar un solo archivo
async def procesar_archivo(file_path):
    print(f"\n[INICIO] Procesando: {file_path}")
    
    # 1. Cargar (Parsear) un solo documento
    # Usamos aload_data (async) para estabilidad
    try:
        # 'documentos' se define aquí
        documentos = await reader.aload_data(file_path)
        if not documentos:
            print(f"  ... Advertencia: LlamaParse no devolvió contenido para {file_path}.")
            return 0
        
        print(f"  ... Parseo (LlamaParse) completado.")
        
        # 2. Ejecutar pipeline (Transformar y Embedir)
        # 'documentos' se usa aquí
        nodes = await pipeline.arun(documents=documentos) 
        
        print(f"[ÉXITO] Documento procesado y almacenado. {len(nodes)} nodos generados.")
        return len(nodes)
    
    except Exception as e:
        # Si 'aload_data' falla, 'documentos' no existe, 
        # pero esta excepción 'e' capturará ese error Y el NameError
        print(f"[ERROR] Falló el procesamiento de {file_path}. Error: {e}")
        return 0

# Bucle principal
async def main_loop():
    total_start_time = time.time()
    nodos_totales = 0
    
    # !!! --- MODIFICACIÓN AQUÍ --- !!!
    # Slicing la lista para empezar desde el archivo 28 (el que falló)
    archivos_pendientes = all_pdf_files[28:] 
    total_archivos = len(all_pdf_files)
    indice_inicio = 28 # Para el logging
    
    for i, file_path in enumerate(archivos_pendientes):
        # Actualizamos el print para que muestre el progreso real
        print(f"\n--- Progreso: Documento {i + indice_inicio + 1} / {total_archivos} ---") 
        # ... (el resto del bucle sigue igual) ...
        file_start_time = time.time()
        
        nodos_generados = await procesar_archivo(file_path)
        nodos_totales += nodos_generados
        
        file_time = time.time() - file_start_time
        print(f"  ... Tiempo del archivo: {file_time:.2f} segundos.")
    
    total_time = time.time() - total_start_time
    print(f"\n--- ¡Ingesta Total Completada! ---")
    print(f"Se procesaron y almacenaron {nodos_totales} nodos en total.")
    print(f"Tiempo total: {(total_time/60):.2f} minutos.")

# Ejecutar el bucle principal en el script
asyncio.run(main_loop())