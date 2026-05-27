import os
import requests
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)

# --- CONFIGURAZIONE ---
# Ollama (Maya)
OLLAMA_API_URL = "https://maya.mynuapp.it/api/generate"
OLLAMA_API_KEY = "47461740de09e3e8644aa6b4a3d4982393910f1f4fc2ca2fc6118b71eae25a6e"
OLLAMA_MODEL = "qwen2.5:1.5b"

# HuggingFace (Fallback)
HF_TOKEN = os.getenv("HF_TOKEN")
hf_client = InferenceClient(token=HF_TOKEN)
HF_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct"
]

ollama_busy = False
ollama_lock = threading.Lock()

def clean_json_response(text):
    return text.replace("```json", "").replace("```", "").strip()

def query_ollama(prompt, system_prompt):
    full_prompt = f"{system_prompt}\n\nUtente: {prompt}\n\nRispondi in formato JSON:"
    payload = {"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False}
    headers = {"Content-Type": "application/json", "X-API-Key": OLLAMA_API_KEY}
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, headers=headers, timeout=25)
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception:
        return None
    return None

def query_huggingface(prompt, system_prompt):
    for model_id in HF_MODELS:
        try:
            response = hf_client.chat_completion(
                model=model_id,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.05
            )
            return response.choices[0].message.content
        except Exception:
            continue
    return None

@app.route('/dashboard-update', methods=['POST'])
def dashboard_update():
    global ollama_busy
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "")
        user_id = data.get("user_id", "")
        nome_attivita = data.get("nome_attivita", "Attività")

        if not user_message:
            return jsonify({"success": False, "reply": "Messaggio vuoto."}), 400

        # PROMPT ORIGINALE RIPRISTINATO
        system_prompt = f"""Sei l'assistente AI del gestionale di {nome_attivita}. L'utente loggato può modificare solo i SUOI prodotti, categorie e menu (utente_id = {user_id}).

Rispondi SEMPRE e SOLO con un oggetto JSON valido. Zero testo fuori dalle graffe.

Azioni disponibili:
- update_product: modifica un prodotto (usa "search_name" se l'ID non è noto)
- update_category: modifica una categoria
- update_menu: modifica un menu
- search_product: cerca prodotto per nome e mostra lista
- unknown: non hai capito

Campi modificabili nel DB:
- prodotti: titolo, ingredienti, prezzo, prezzo_scontato, visibile, prodotto_fresco, prodotto_congelato, senza_glutine, senza_lattosio, note, disponibile_senzaGlutine, disponibile_senzaLattosio, prezzo_senzaGlutine, prezzo_senzaLattosio, categoria_id
- categorie: titolo, descrizione, visibile, colore, ordinamento
- menu: titolo, prezzo_fisso, visibile

Regole:
- Per menu e categorie usa "titolo" e MAI "nome".
- "visibile" accetta SOLO 1 o 0."""

        # 1. Tenta Ollama
        used_ollama = False
        with ollama_lock:
            if not ollama_busy:
                ollama_busy = True
                used_ollama = True

        if used_ollama:
            res = query_ollama(user_message, system_prompt)
            with ollama_lock:
                ollama_busy = False
            if res:
                return jsonify({"success": True, "raw_text": clean_json_response(res)})
            else:
                with ollama_lock:
                    ollama_busy = False

        # 2. Fallback su HF
        res = query_huggingface(user_message, system_prompt)
        if res:
            return jsonify({"success": True, "raw_text": clean_json_response(res)})

        return jsonify({"success": False, "reply": "Sistemi AI non disponibili."}), 503

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
