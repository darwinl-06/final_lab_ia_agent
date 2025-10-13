# rag/ingest.py — Ingesta simple desde ./data (PDF + XLS) a Chroma
from pathlib import Path
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

DATA_DIR = Path("data")
PERSIST_DIR = "./chroma"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_pdf(name: str):
    path = DATA_DIR / name
    if not path.exists():
        return []
    docs = PyPDFLoader(str(path)).load()
    for d in docs:
        d.metadata["source"] = name
    return docs

def load_excel_rows(filename: str, sheet: str, row_to_text):
    path = DATA_DIR / filename
    if not path.exists():
        return []
    # pandas detecta engine según extensión; para .xls usa xlrd
    df = pd.read_excel(path, sheet_name=sheet)
    docs = []
    for _, row in df.iterrows():
        text = row_to_text(row)
        docs.append(Document(
            page_content=text,
            metadata={"source": filename, "sheet": sheet}
        ))
    return docs

def row_to_text_orders(row):
    return (
        f"PEDIDO\n"
        f"TRACKING: {row.get('tracking', '')}\n"
        f"STATUS: {row.get('status', '')}\n"
        f"ETA: {row.get('eta', '')}\n"
        f"CARRIER: {row.get('carrier', '')}\n"
        f"TRACKING_URL: {row.get('tracking_url', '')}\n"
        f"LAST_UPDATE: {row.get('last_update', '')}\n"
        f"DELAY_REASON: {row.get('delay_reason', '')}\n"
    )

def row_to_text_products(row):
    return (
        f"PRODUCTO\n"
        f"SKU: {row.get('sku', '')}\n"
        f"NOMBRE: {row.get('name', '')}\n"
        f"CATEGORIA: {row.get('category', '')}\n"
        f"RETORNABLE: {row.get('returnable', '')}\n"
    )

def build_index():
    docs = []
    # PDFs
    docs += load_pdf("returns_policy.pdf")
    docs += load_pdf("faq.pdf")

    docs += load_excel_rows("orders.xls",   sheet="orders",   row_to_text=row_to_text_orders)
    docs += load_excel_rows("products.xls",  sheet="products", row_to_text=row_to_text_products)

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"[OK] Index construido en {Path(PERSIST_DIR).resolve()} con {len(chunks)} chunks.")

if __name__ == "__main__":
    build_index()
