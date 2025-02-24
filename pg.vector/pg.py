# Import necessary modules and classes
from agno.agent import Agent  # Agent class for managing interactions
from agno.models.openai import OpenAIChat  # OpenAIChat model for chat interactions
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader  # PDF knowledge base and reader
from agno.vectordb.pgvector import PgVector, SearchType  # Vector database and search types
from agno.models.ollama import Ollama  # Ollama model for AI interactions
import sqlalchemy  # SQL toolkit for Python

# Database URL for connecting to the vector database
db_url = ""

# Initialize the PDF knowledge base with the specified path and vector database
knowledge_base = PDFKnowledgeBase(
    path="244 Submission.pdf",  # Path to the PDF file
    vector_db=PgVector(table_name="recipes", db_url=db_url, search_type=SearchType.hybrid),  # Vector DB configuration
)

# Load the knowledge base: Comment out after first run
knowledge_base.load(recreate=True, upsert=True)  # Load the knowledge base, recreating if necessary

# Create an instance of the Agent with the specified model and knowledge base
agent = Agent(
    model=Ollama(id="llama3.2"),  # Specify the model to use
    knowledge=knowledge_base,  # Attach the knowledge base to the agent
    # Add a tool to read chat history.
    read_chat_history=True,  # Enable reading of chat history
    show_tool_calls=True,  # Show tool calls in the output
    markdown=True,  # Enable markdown formatting in responses
    # debug_mode=True,  # Uncomment to enable debug mode
)

# Print a response from the agent with a specific query
agent.print_response("why this project need ", stream=True)  # Query the agent for a response
