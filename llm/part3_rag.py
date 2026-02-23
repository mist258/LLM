import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"


# STEP 1: Load the document and split it into paragraphs
def load_chunks(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
        # Filter out empty paragraphs
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return chunks


# STEP 2: Searching for the relevant fragment
def find_best_chunk(question: str, chunks: list[str]) -> str:
    question_words = set(question.lower().split())

    best_chunk = ""
    best_score = 0

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words & chunk_words)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk

# STEP 3: Two query options — without and with context
def ask(system_prompt: str, question: str) -> str:
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def ask_without_context(question: str) -> str:
    return ask("Answer the questions as critically and analytically as possible", question)


def ask_with_context(question: str, context: str) -> str:
    system = f"""Answer the questions only based on the provided context.
    If the answer is not in the context, explicitly say so.
    CONTEXT: {context}"""
    return ask(system, question)


chunks = load_chunks("document.txt")
print(f"The document is split into {len(chunks)} paragraphs\n")

question = "Key difference between of the TCP/IP and OSI model?"

best = find_best_chunk(question, chunks)
print(f"The most relevant fragment:\n{best}\n")

print("=== WITHOUT CONTEXT ===")
print(ask_without_context(question))

print("\n=== WITH CONTEXT ===")
print(ask_with_context(question, best))