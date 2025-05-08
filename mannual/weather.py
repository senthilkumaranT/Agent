from langchain_community.utilities import OpenWeatherMapAPIWrapper

def weather_call(user_input: str):
  # Importing necessary libraries

  weather = OpenWeatherMapAPIWrapper(openweathermap_api_key="")  

  weather_call = weather.run(user_input)
  return weather_call

