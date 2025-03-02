import os
from agno.agent import Agent, RunResponse
from agno.models.openrouter import OpenRouter
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.playground import Playground, serve_playground_app
from agno.models.ollama import Ollama
from agno.tools.zoom import ZoomTools

import streamlit as st

# Streamlit input for API key
api_key = st.text_input("Enter your OpenRouter API key: ")

# Web search agent 
web_search_agent = Agent(
    name="web search agent",
    role="search the web for the information",
    model=OpenRouter(id="gpt-4o-mini", api_key=api_key),  # Use the OpenRouter API key from user input
    tools=[DuckDuckGoTools()],
    instructions=["always include sources"],
    show_tool_calls=True,
    markdown=True
)

# Finance agent
finance_agent = Agent(
    name="Finance AI agent",
    model=OpenRouter(id="gpt-4o-mini", api_key=api_key),  # Use the OpenRouter API key from user input
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["use tables to display data"],
    show_tool_calls=True,
    markdown=True
)

multi_agent = Agent(
    team=[finance_agent, web_search_agent],
    name="multi agent",
    model=OpenRouter(id="gpt-4o-mini", api_key=api_key),  # Use the OpenRouter API key from user input
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["always include sources","use tables to display data"],
    show_tool_calls=True,
    markdown=True
)

st.title("Financial Agent Query")

question = st.text_input("Enter your question: ")
if st.button("Submit"):
    response = multi_agent.run(question)
    st.write(response.content)
