import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

task = """It’s suitable for a gas boiler, laptop, lighting, and charging any gadgets. 
        It’s not suitable for devices that consume more than 500W. 
        Overall, I’m satisfied with the station, although its capacity isn’t very large, 
        and that should be taken into account."""

def ask(system_prompt, user_message):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content

# role prompting

role_prompt = """You are an experienced review analyst for an e-commerce platform.
Your task is to accurately determine the sentiment of reviews and identify key topics.
Always respond in a structured format: sentiment, rating from 1 to 5, key topics."""

result_role = ask(role_prompt, f"Analyze answer: {task}")

# Few-shots

few_shot_prompt = """Determine the sentiment of the review. Here are some examples:

Review: "The charging station is well-packaged and well-built. It’s compact and has a strong strap for carrying. 
        The front panel features a light with three modes, which is very convenient."
Answer: POSITIVE — The customer is satisfied with the quality and performance.

Review: "While it was charged, everything worked fine. But as soon as the battery ran out and I put it on charge, 
        something inside sparked and smoke appeared. 
        After that, it went completely dead and now it neither turns on nor charges."
Answer: NEGATIVE — problems with the product’s quality; the charging station won’t turn on anymore.

Review: "I bought this product. It doesn’t sync via Bluetooth with my phone. 
        We downloaded the app from the official website and registered, but nothing changed. 
        The phone still doesn’t detect the device. 
        However, the charging station works as described."
Answer: MIXED — there are advantages, but there is a significant drawback."""

result_few = ask(few_shot_prompt, f"Review: {task}")

# Chain-of-Thought
cot_prompt = """Perform a step-by-step analysis of the product review:

1. First, list all the positive aspects mentioned by the reviewer.
2. Then, list all the negative aspects.
3. Compare their weight and importance to the customer.
4. Only after that, draw a conclusion about the overall sentiment."""

result_cot = ask(cot_prompt, f"Review: {task}")

print("**** ROLE PROMPTING ****")
print(result_role)
print("\n**** FEW-SHOT ****")
print(result_few)
print("\n**** CHAIN-OF-THOUGHT ****")
print(result_cot)