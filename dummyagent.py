import os
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

# Set up the Hugging Face token
os.environ["HF_TOKEN"] =""

# Define the system prompt
SYSTEM_PROMPT = "You are a helpful assistant."
city = input("Enter the city to see the weather: ")

# Prepare messages for the model
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"What's the weather in {city}?"},  # Updated to use the user input
]

# Attempt to load the tokenizer and handle potential errors
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
except Exception as e:
    print()

# Initialize the InferenceClient
try:
    client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud")
except Exception as e:
    print(f"Error initializing InferenceClient: {e}")

# Create the prompt
prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use : 

{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. 
<|eot_id|><|start_header_id|>user<|end_header_id|>
What's the weather in London ?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Action:
```
{
  "action": "get_weather",
  "action_input": {"location": {"type": "string", "value": "London"}
}

Thought: I will check the weather in London.
Observation:the weather in London is sunny with low temperatures. """

# Generate output from the model

output = client.text_generation(
    prompt,
    max_new_tokens=200,
)
print(output)

