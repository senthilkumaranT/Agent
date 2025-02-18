import os
import typer
from rich.prompt import Prompt
from typing import Optional
from agno.agent import Agent
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.qdrant import Qdrant
from agno.vectordb.pgvector import PgVector
from sqlalchemy.exc import NoSuchModuleError

from dotenv import load_dotenv
load_dotenv()

os.environ["OPEN_ROUTER_API_KEY"] = os.getenv("OPEN_ROUTER_API_KEY")
api_key = os.getenv("QDRANT_API_KEY")
db_url = os.getenv("QDRANT_URL")  # Updated to use environment variable for Qdrant URL

vector_db = Qdrant(
    collection="pdf_documents",
    url=db_url,
    api_key=api_key,
)

knowledge = PDFKnowledgeBase(
    path="Pacemakers_and_implantable_cardiac_defib.pdf",
    vector_db=vector_db,
    reader=PDFReader(chunk=True),
)

knowledge.load()



def qdrant_agent(user: str = "user"):
    run_id: Optional[str] = None

    agent = Agent(
        run_id=run_id,
        user_id=user,
        knowledge=knowledge,
        tool_calls=True,
        use_tools=True,
        show_tool_calls=True,
        debug_mode=True,
    )

    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        agent.print_response(message)

if __name__ == "__main__":
    # Comment out after first run
    knowledge.load(recreate=True, upsert=True)
    typer.run(qdrant_agent)