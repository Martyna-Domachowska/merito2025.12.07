from transformers import pipeline
import os
import torch as _torch

def ensure_model_local(model_name: str, model_dir: str) -> str:
    """
    Funkcja:
    - jeśli lokalny katalog modelu istnieje i nie jest pusty -> zwraca ścieżkę do modelu
    - w przeciwnym razie -> zwraca nazwę modelu (transformers pobierze go automatycznie)
    """
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        return model_dir
    return model_name


model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model_dir = os.path.join(os.path.dirname(__file__), "models", "distilbert-sst2")

local_model_path = ensure_model_local(model_name, model_dir)

# NIE ustawiamy framework — transformers sam wykrywa czy masz PyTorch, czy TensorFlow
classifier = pipeline(
    "sentiment-analysis",
    model=local_model_path,
    tokenizer=local_model_path
)

print(classifier("James bond is a woman!"))
print(classifier("The movie was average but acceptable. Not good, not bad."))

from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="bardsai/twitter-sentiment-pl-base",
    tokenizer="bardsai/twitter-sentiment-pl-base"
)

print(classifier("Kocham ten produkt!"))
print(classifier("Nie podoba mi się jakość — jestem zawiedziony."))

