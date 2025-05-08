from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.models.openrouter import OpenRouter


agent = Agent(
     name="finance agent",
     tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
     description="you are a expert in the financial stock price you want to give it in the proper documentation ",
     model=OpenRouter(id="gpt-4o-mini", api_key=""),
     show_tool_calls=True,
)

agent.print_response("what is the current stoce price of nvidia")
