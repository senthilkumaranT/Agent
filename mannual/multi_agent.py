from chat import chat, completion
from pydantic import BaseModel

from retri import retrieve_chunks
from weather import weather_call

class Identifier(BaseModel):
  is_search: bool
  is_weather: bool

class Weather(BaseModel):
  place: str

WEATHER_PROMPT = """
  You are a agent good in finding places from the user's query.
  User's query: {user_input}
  Place:
"""

IDENTIFIER_PROMPT = """
  User's query: {user_input}

  You are an agent specialized in query classification.
  Your task is to determine if the user's query requires an rag search.
  
  Guidelines:
  - Classify as search true if the query is general knowledge (e.g. "Who is...", "What is...", "When did...")
  - Classify as search if the query asks about facts, history, or current events
  - Do not classify as search if the query is about weather or local conditions
  - Classify as weather if the query asks about temperature, precipitation, or weather conditions
  - Classify as weather if the query mentions forecast, weather, rain, sun, etc.
  - Classify as weather if the query asks about current or future weather conditions for a location
  
  Analyze the query and determine if it requires searching.
"""

def mood_finder(user_input: str):
  from openai import OpenAI
  
  client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="")
  
  response = client.beta.chat.completions.parse(
      model="openai/gpt-4o-mini",
      messages=[
        {
          'role': 'user',
          'content': user_input,
        }
      ],
      response_format=Identifier,
  )
  
  filtered_response = Identifier.model_validate_json(response.choices[0].message.content)
  return filtered_response

query = "tell about the madurai weather"

identifier = mood_finder(IDENTIFIER_PROMPT.format(user_input=query))
print(identifier)

if identifier.is_search:
  rag_response = chat(query)

if identifier.is_weather:
  weather_response = weather_call(user_input="tokyo")
  completion(question=query, output=weather_response)
