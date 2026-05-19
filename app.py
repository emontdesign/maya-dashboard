import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import sys

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct", # Modello più potente per gestire menu complessi
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct"
]

@app.route('/', methods=['GET'])
def home():
    return "Maya AI (Human & Nested Menu) Online", 200

@app.route('/ask', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "")
        nome_rist = data.get("nome", "il ristorante")
        giorno_oggi = data.get("giorno_settimana", "oggi")
        
        # --- 1. ESTRAZIONE CHIRURGICA DEL MENU ---
        menu_input = data.get("menu", [])
        if isinstance(menu_input, str):
            try: menu_input = json.loads(menu_input)
            except: menu_input = []
        
        # Se il JSON ha la chiave "data", prendiamo quella
        menu_list = menu_input.get("data", []) if isinstance(menu_input, dict) else menu_input

        menu_narrativo = ""
        if isinstance(menu_list, list):
            for m in menu_list:
                # Titolo del Menu (Pranzo, Cena, ecc.)
                menu_narrativo += f"\n### {m.get('titolo')} ###\n"
                if m.get('prezzo_fisso', 0) > 0:
                    menu_narrativo += f"(Prezzo fisso: {m.get('prezzo_fisso')}€)\n"
                
                for cat in m.get('categorie', []):
                    prodotti = cat.get('prodotti', [])
                    if prodotti: # Regola: Solo categorie con prodotti
                        menu_narrativo += f"  - Categoria {cat.get('titolo')}:\n"
                        for p in prodotti:
                            nota = p.get('note', "")
                            if not nota or nota.lower() == "nota prodotto": nota = ""
                            prezzo = p.get('prezzo_scontato') if p.get('prezzo_scontato') else p.get('prezzo')
                            menu_narrativo += f"    * {p.get('titolo')} ({prezzo}€) {'- ' + nota if nota else ''}\n"

        # --- 2. ESTRAZIONE ORARI ---
        hours_raw = data.get("hours", {})
        if isinstance(hours_raw, str):
            try: hours_raw = json.loads(hours_raw)
            except: hours_raw = {}
        
        orari_oggi = "purtroppo non li ho caricati bene"
        is_open_now = False
        if isinstance(hours_raw, dict):
            status = hours_raw.get("status", {})
            is_open_now = status.get("is_open", False)
            idx = status.get("day_index")
            schedule = hours_raw.get("schedule", [])
            if idx is not None and len(schedule) > idx:
                turni = [f"dalle {t.get('apertura')[:5]} alle {t.get('chiusura')[:5]}" 
                         for t in schedule[idx] if t.get('is_closed') == 0]
                orari_oggi = " e ".join(turni) if turni else "oggi siamo chiusi"

        # --- 3. SYSTEM INSTRUCTIONS (Maya Persona) ---
        system_instructions = f"""
Sei Maya, l'assistente virtuale di {nome_rist}. Rispondi in italiano con calore e naturalezza.

PERSONALITÀ:
- Sei solare, amichevole e pronta ad aiutare. Non sei un robot.
- Rispondi come se parlassi a un amico che entra ora nel locale.
- Se ti salutano, ricambia il saluto con gioia e chiedi cosa puoi fare. NON dare orari o menu subito.

CONTESTO (USA SOLO SE PERTINENTE):
- Giorno: {giorno_oggi}
- Stato Attuale: {"Aperti e pronti ad accoglierti!" if is_open_now else "Chiusi al momento."}
- Orari di oggi: {orari_oggi}.
- Proposte del Menu: {menu_narrativo if menu_narrativo else "Stiamo aggiornando le specialità del giorno."}

REGOLE DI RISPOSTA:
1. NO ELENCHI FREDDI: Se chiedono il menu, presentalo con entusiasmo (es: "Oggi a pranzo abbiamo una deliziosa Pasta al Pomodoro...").
2. NO "NOTA PRODOTTO": È severamente vietato dire "Nota prodotto". Se la nota è vuota, ignora.
3. NO RIGIDITÀ: Non ripetere "Oggi è martedì". Se chiedono gli orari, di' semplicemente quando siete aperti.
4. ORDINI: Se chiedono di ordinare, dì che per ora sei qui per dare info, ma non puoi ancora prendere ordini in chat.
5. NO RIPETIZIONI: Quando fornisci una risposta bada bene a non ripetere una parola a meno che non abbia un senso compiuto (ed: "Ciao ciao! ..").
"""

        for model_id in MODELS:
            try:
                response = client.chat_completion(
                    model=model_id,
                    messages=[{"role": "system", "content": system_instructions}, {"role": "user", "content": query}],
                    max_tokens=500,
                    temperature=0.7 # Temperatura ideale per una conversazione umana
                )
                return jsonify({"success": True, "reply": response.choices[0].message.content})
            except: continue

        return jsonify({"success": False, "reply": "Ops, mi sono un attimo distratta! Riprova? 🍕"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
