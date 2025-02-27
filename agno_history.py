from agno.agent import Agent

from rich.pretty import pprint
from agno.models.ollama import Ollama  


agent = Agent(
    model=Ollama(id="llama3.2:latest"),
    # Set add_history_to_messages=true to add the previous chat history to the messages sent to the Model.
    add_history_to_messages=True,
    # Number of historical responses to add to the messages.
    num_history_responses=3,
    description="You are a helpful assistant that always responds in a polite, upbeat and positive manner.",
)

# -*- Create a run
agent.print_response("Share a 2 sentence horror story", stream=True)

# -*- Print the messages in the memory
agent.print_response("tell me a  joke", stream=True)

# -*- Ask a follow up question that continues the conversation
agent.print_response("summarize the conversation in 2 sentences", stream=True)

