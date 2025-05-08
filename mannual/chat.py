from openai import OpenAI
from retri import retrieve_chunks

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="")

def chat(question: str):
    output = retrieve_chunks(question)

    prompt = f"""
        question: {question}
        output: {output[0]['text']}
        answer the question based on the output
    """

    print(prompt)

    stream = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "system",
                "content": """You are a helpful assistant that can answer questions about the world. your name is jake you always answer in a funny way with emojies like a genz in America""",
            },
        ],
        stream=True,
    )

    for chunk in stream:
        # print(chunk)
        print(chunk.choices[0].delta.content, end="", flush=True)

def completion(question: str, output: str):
    prompt = f"""
        question: {question}
        output: {output}
        answer the question based on the output
    """

    print(prompt)

    stream = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "system",
                "content": """You are a helpful assistant that can answer questions about the world. your name is jake you always answer in a funny way with emojies like a genz in America""",
            },
        ],
        stream=True,
    )

    for chunk in stream:
        # print(chunk)
        print(chunk.choices[0].delta.content, end="", flush=True)
