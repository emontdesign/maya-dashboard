import os
import json
import httpx
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")
OLLAMA_TOKEN = os.getenv("MY_TOKEN")  # 🔥 Token per Ollama
client = InferenceClient(token=HF_TOKEN)

# Modelli HuggingFace (fallback)
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct"
]

# Configurazione Ollama (VPS2)
OLLAMA_URL = "https://maya.mynuapp.it/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"


# -----------------------------
# FUNZIONE: chiamata a OLLAMA
# -----------------------------
async def call_ollama(prompt: str):
    try:
        async with httpx.AsyncClient(timeout=6.0, verify=False) as client_ollama:
            response = await client_ollama.post(
                OLLAMA_URL,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": OLLAMA_TOKEN  # 🔥 Token inserito qui
                },
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", None)
    except Exception:
        return None  # Ollama non disponibile → fallback HF


# -----------------------------
# FUNZIONE: fallback HuggingFace
# -----------------------------
def call_huggingface(system_prompt, user_message):
    raw_response = None

    for model_id in MODELS:
        try:
            response = client.chat_completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=300,
                temperature=0.05
            )
            raw_response = response.choices[0].message.content
            if raw_response:
                break
        except:
            continue

    return raw_response


# -----------------------------
# ENDPOINT PRINCIPALE
# -----------------------------
@app.route('/dashboard-update', methods=['POST'])
async def dashboard_update():
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "")
        user_id = data.get("user_id", "")
        nome_attivita = data.get("nome_attivita", "Attività")

        if not user_message:
            return jsonify({"success": False, "reply": "Messaggio vuoto."}), 400

        # SYSTEM PROMPT
        system_prompt = f"""Sei l'assistente AI del gestionale di {nome_attivita}. L'utente loggato può modificare solo i SUOI prodotti, categorie e menu (utente_id = {user_id}).

Rispondi SEMPRE e SOLO con un oggetto JSON valido. Zero testo fuori dalle graffe.

Azioni disponibili:
- update_product: modifica un prodotto (usa "search_name" se l'ID non è noto)
- update_category: modifica una categoria
- update_menu: modifica un menu
- search_product: cerca prodotto per nome e mostra lista
- unknown: non hai capito

Campi modificabili nel DB (usa esattamente questi nomi nei "fields"):
- prodotti: titolo, ingredienti, prezzo, prezzo_scontato, visibile, prodotto_fresco, prodotto_congelato, senza_glutine, senza_lattosio, note, disponibile_senzaGlutine, disponibile_senzaLattosio, prezzo_senzaGlutine, prezzo_senzaLattosio, categoria_id
- categorie: titolo, descrizione, visibile, colore, ordinamento
- menu: titolo, prezzo_fisso, visibile

Regole speciali per i campi:
- Per i menu e le categorie si usa sempre il campo "titolo" e MAI "nome".
- I campi "visibile" (presenti in prodotti, categorie e menu) accettano SOLO 1 (attivo) o 0 (disattivo).

Esempi:
Utente: "Cambia prezzo della pizza margherita a 8 euro"
{{"action":"update_product","search_name":"pizza margherita","fields":{{"prezzo":8.00}}}}

Utente: "Rinomina categoria Primi in Primi Piatti"
{{"action":"update_category","search_name":"Primi","fields":{{"titolo":"Primi Piatti"}}}}

Utente: "Nascondi la categoria Dolci dal menu pubblico"
{{"action":"update_category","search_name":"Dolci","fields":{{"visibile":0}}}}"""

        # -----------------------------
        # 1️⃣ PROVA OLLAMA (PRINCIPALE)
        # -----------------------------
        ollama_reply = await call_ollama(
            f"{system_prompt}\nUtente: {user_message}"
        )

        if ollama_reply:
            return jsonify({
                "success": True,
                "provider": "ollama",
                "raw_text": ollama_reply
            })

        # -----------------------------
        # 2️⃣ FALLBACK HUGGINGFACE
        # -----------------------------
        hf_reply = call_huggingface(system_prompt, user_message)

        if hf_reply:
            return jsonify({
                "success": True,
                "provider": "huggingface",
                "raw_text": hf_reply
            })

        # -----------------------------
        # 3️⃣ TUTTO OFFLINE
        # -----------------------------
        return jsonify({
            "success": False,
            "reply": "Ops, tutti i moduli AI sono momentaneamente offline. Riprova tra poco! 🍕"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
