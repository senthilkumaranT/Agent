import os
import typer 
from typing import Optional,List
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.pgvector import PgVector
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.embedder.ollama import OllamaEmbedder
from dotenv import load_dotenv

load_dotenv()


os.environ["AGNO_API_KEY"] = os.getenv("AGNO_API_KEY")
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")


db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

pdf_knowledge_base = PDFKnowledgeBase(
    path="data/pdfs",
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url=db_url,
    ),
    reader=PDFReader(chunk=True),
    embedding_model=OllamaEmbedder(id="lamma3.2:latest"),
)


knowledge_base = PDFUrlKnowledgeBase(
    vector_db=PgVector(table_name="pdf_documents", db_url=db_url, search_type=SearchType.hybrid)
)





