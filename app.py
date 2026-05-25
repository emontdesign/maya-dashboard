import os
import requests
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configurazione API Ollama (Maya)
API_URL = "https://maya.mynuapp.it/api/generate"
API_KEY = "47461740de09e3e8644aa6b4a3d4982393910f1f4fc2ca2fc6118b71eae25a6e"
MODEL_NAME = "qwen2.5:1.5b"

@app.route('/dashboard-update', methods=['POST'])
def dashboard_update():
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "")
        user_id = data.get("user_id", "")
        nome_attivita = data.get("nome_attivita", "Attività")

        if not user_message:
            return jsonify({"success": False, "reply": "Messaggio vuoto."}), 400

        # System Prompt come definito da te
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

        # Costruzione del prompt per il modello
        # Concateniamo System e User per il formato 'generate' di Ollama
        full_prompt = f"{system_prompt}\n\nUtente: {user_message}\n\nRispondi in formato JSON:"

        payload = {
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "stream": False # Necessario per ricevere la risposta completa in un blocco solo
        }

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        }

        # Chiamata all'API
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            raw_response = result.get("response", "").strip()
            
            # Pulisce eventuali caratteri di formattazione Markdown (se il modello li aggiunge)
            raw_response = raw_response.replace("```json", "").replace("
```", "").strip()
            
            return jsonify({"success": True, "raw_text": raw_response})
        else:
            return jsonify({"success": False, "reply": "Errore di connessione al motore AI.", "details": response.text}), 500

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
