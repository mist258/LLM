import os
import json
from groq import Groq
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "mixed", "neutral"] = Field(
        description="Overall sentiment of the review"
    )
    score: float = Field(
        ge=1.0, le=5.0,
        description="Score from  1.0 to 5.0"
    )
    positive_aspects: list[str] = Field(
        description="List of positive aspects"
    )
    negative_aspects: list[str] = Field(
        description="List of negative aspects"
    )
    summary: str = Field(
        max_length=200,
        description="A brief one-sentence summary"
    )

    @field_validator("score")
    @classmethod
    def validate_score_consistency(cls, v):
        return round(v, 1)

def analyze_review(review_text: str) -> SentimentAnalysis:

    system_prompt = f"""You analyze product reviews.
        Respond only with valid JSON without any additional text.
        The JSON must have the following structure:
        {json.dumps(SentimentAnalysis.model_json_schema(), ensure_ascii=False, indent=2)}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Проаналізуй відгук: {review_text}"}
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw_json = response.choices[0].message.content

    try:
        result = SentimentAnalysis.model_validate_json(raw_json)
        return result
    except Exception as e:
        print(f"Validation error: {e}")
        print(f"Row JSON from model: {raw_json}")
        raise

review = """ It’s suitable for a gas boiler, laptop, lighting, and charging any gadgets. 
        It’s not suitable for devices that consume more than 500W. 
        Overall, I’m satisfied with the station, although its capacity isn’t very large, 
        and that should be taken into account."""

result = analyze_review(review)

print(f"Sentiment: {result.sentiment}")
print(f"Score: {result.score}/5.0")
print(f"Advantages: {result.positive_aspects}")
print(f"Disadvantages: {result.negative_aspects}")
print(f"Conclusion: {result.summary}")

print("\nJSON for API:")
print(result.model_dump_json(indent=2))