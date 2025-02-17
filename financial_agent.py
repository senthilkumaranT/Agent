import os
from agno.agent import Agent, RunResponse
from agno.models.openrouter import OpenRouter
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.playground import Playground, serve_playground_app


# Load API keys from environment variables
AGNO_API_KEY = os.getenv("AGNO_API_KEY")  # Updated to use a proper environment variable name
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Updated to use a proper environment variable name

# Web search agent 
web_search_agent = Agent(
    name="web search agent",
    role="search the web for the information",
    model=OpenRouter(id="deepseek/deepseek-r1-distill-qwen-1.5b", api_key=OPENROUTER_API_KEY),  # Use the OpenRouter API key
    tools=[DuckDuckGoTools()],
    instructions=["always include sources"],
    show_tool_calls=True,
    markdown=True
)

# Finance agent
finance_agent = Agent(
    name="Finance AI agent",
    model=OpenRouter(id="deepseek/deepseek-r1-distill-qwen-1.5b", api_key=OPENROUTER_API_KEY),  # Use the OpenRouter API key
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["use tables to display data"],
    show_tool_calls=True,
    markdown=True
)

app = Playground(agents=[finance_agent, web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)