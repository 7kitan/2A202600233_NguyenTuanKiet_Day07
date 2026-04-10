import os

from dotenv import load_dotenv

from src.chunking import compute_similarity
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    OPENAI_EMBEDDING_MODEL,
    OpenAIEmbedder,
)

load_dotenv(override=False)
provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
embedder = OpenAIEmbedder(
    model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
)

sentence_pairs = [
    ("The cat sat on the mat.", "The cat is on the mat."),
    ("I love programming Python.", "Python is a type of snake."),
    ("The weather is sunny today.", "It is a bright, sunny day."),
    ("Apples are my favorite fruit.", "I really enjoy eating apples."),
    ("Democracy is a form of government.", "Pizza is a popular Italian food."),
]


def test_compute_similarity():
    print(f"{'Pair':<5} | {'Score':<10} | {'Note'}")
    print("-" * 45)

    for i, (sent_a, sent_b) in enumerate(sentence_pairs):
        v_a = embedder(sent_a)
        v_b = embedder(sent_b)

        score = compute_similarity(v_a, v_b)

        print(f"{i:<5} | {score:<10.4f} | {sent_a} | {sent_b}")


if __name__ == "__main__":
    test_compute_similarity()

""" output
Pair  | Score      | Note
---------------------------------------------
0     | 0.8666     | The cat sat on the mat. | The cat is on the mat.
1     | 0.5119     | I love programming Python. | Python is a type of snake.
2     | 0.7823     | The weather is sunny today. | It is a bright, sunny day.
3     | 0.7169     | Apples are my favorite fruit. | I really enjoy eating apples.
4     | 0.1103     | Democracy is a form of government. | Pizza is a popular Italian food.
"""
