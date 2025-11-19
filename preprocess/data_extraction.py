import os
import csv
import random

import requests
from tqdm import tqdm

# --------------------------
# CONFIGURAZIONE
# --------------------------
CARTELLA_INPUT = "./USoR/bronze"
FILE_OUTPUT = "model_output.csv"
MODELLO_OLLAMA = "llama3.1"
# --------------------------


def chiedi_ollama(prompt: str, modello: str):
    """
    Manda una query a Ollama e restituisce solo il testo generato.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": modello,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def unisci_file_txt(cartella: str):
    """
    Legge tutti i .txt nella cartella e restituisce una lista di righe.
    Gestisce file non UTF-8 senza crash.
    """
    righe_totali = []

    for nomefile in os.listdir(cartella):
        if nomefile.endswith(".txt"):
            percorso = os.path.join(cartella, nomefile)

            try:
                with open(percorso, "r", encoding="utf-8") as f:
                    righe = f.readlines()
            except UnicodeDecodeError:
                try:
                    with open(percorso, "r", encoding="cp1252") as f:
                        righe = f.readlines()
                except UnicodeDecodeError:
                    with open(percorso, "r", encoding="latin1", errors="replace") as f:
                        righe = f.readlines()

            righe = [r.strip() for r in righe if r.strip()]
            righe_totali.extend(righe)

    return righe_totali


def main():
    righe = unisci_file_txt(CARTELLA_INPUT)

    if not righe:
        print("Nessun file .txt trovato o nessuna riga valida.")
        return

    print(f"Totale righe caricate: {len(righe)}")
    random.shuffle(righe)

    with open(FILE_OUTPUT, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["user_story", "requisito"])  # <-- due colonne

        for riga in tqdm(righe, desc="Estrazione requisiti"):
            prompt = f"""
            Generate a software non functional requirement on the system performance from the following user story.
            The output MUST be only in the form: req: <requirement>
            No additional text. No explanations. No comments.


            Sentence: "{riga}"
            """

            try:
                requisito = chiedi_ollama(prompt, MODELLO_OLLAMA)
            except Exception as e:
                requisito = f"ERROR"

            writer.writerow([riga, requisito])


if __name__ == "__main__":
    main()