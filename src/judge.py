"""LLM-as-judge relevance scoring (1-5) for retrieved chunks, since no human grader is available."""
import json
import os

from openai import OpenAI

CHAT_MODEL = "gpt-4o-mini"

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

JUDGE_PROMPT = """You are grading a medical information retrieval system.

Query: {query}

Retrieved passages:
{context}

Rate how relevant these retrieved passages are for answering the query, on a scale of 1-5:
5 = passages directly and completely answer the query
3 = passages are topically related but miss the specific answer
1 = passages are unrelated to the query

Respond with ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""


def score_relevance(query, chunks):
    context = "\n---\n".join(chunks)
    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(query=query, context=context)}],
    )
    data = json.loads(response.choices[0].message.content)
    return data["score"], data["reason"]
