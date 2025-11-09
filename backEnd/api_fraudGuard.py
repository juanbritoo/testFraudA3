from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS


'''
Objetivo: Criar um backEnd com API para fazaer todo o tramento é lógica do Projeto.
Neste back estamos utilizando o o Flask, um framework que que permite criar APIs de forma simples.
Core de todo o backEnd: Foi treinado todos os modelos de IA, porém devido a poucas evidências de fraude, os modelos ficaram um pouco desbalanceaods
aprendendo mais sobre transaçãoes normais, do que transações fraudulentas, o nome disso é Classe Desbalanceada!
Para corrigir esse problema, uma possível solução poderia utilizar o smote Balance para criar mais dados sintéticos, melhorando as perfomances dos 
modelos, porém, traria um retrabalho em treinar todos os modelos novamente. Para contornar essa situação mantendo o treinamento de modelo atual
foi decidido aplicar algumas regras!
Essa camada foi adicionada para amplificar a probabilidade de risco em algumas situações, aplicando um peso maior. Ou seja,
dependendo do input do usuário, ele ira pegar esse dado e amplificar a % de fraude dentro de uma regra de if e elif. Ex.: se o input do usuário passar de 
3k e a categoria for travel, utilizando como ex  8.000, ele vai aplicar um cálculo e dividir esse valor por 10.000, para colocar em uma escala de 0 a 1,
depois irá multiplicar por 15, aplicando um peso a situação, o que iria trazer 12% ((8000/10000)* 15). Ou seja, aumentou em 12% o risco de fraude. Dentro
do código, essa situação está limitada a 25%, ou seja, não pode ultrapassar esse valor.
Para as horas, se for de madrugada aplica 20%, ou em algum estado específico, ou parte do dia ele vai adicionando cada vez mais % através dessas condições
OBS: olhe o código para entender as situações
'''

app = Flask(__name__)
CORS(app)

model_path = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmosPython\treinandoModelos\modelo_final_ensemble_ponderado.pkl"
scaler_path = r"C:\Users\juan_\Downloads\testFraudA3-main\pre_processamento_baseDeDados\scaler_final.joblib"
mapeamentos_path = r"C:\Users\juan_\Downloads\testFraudA3-main\backEnd\mapeamentos_target_encoding.joblib"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
mapeamentos = joblib.load(mapeamentos_path)

def calcular_risco_amplificado(transaction_data, probability_base):
    risco = probability_base * 100
    categoria = transaction_data["category"]
    valor = transaction_data["amt"]
    hora = transaction_data["hour"]
    estado = transaction_data["state"]
    parte_dia = transaction_data["part_of_day"]
    dia_semana = transaction_data["day_of_week"]

    if categoria in ["travel", "shopping_net", "misc_pos"]:
        if valor > 3000:
            risco += min((valor / 10000) * 15, 25)
    else:
        if valor > 10000:
            risco += 10

    if hora in [0, 1, 2, 3, 4, 5]:
        risco += 20
    elif hora in [22, 23]:
        risco += 10

    if dia_semana in [6, 7]:
        risco += 5

    if categoria in ["travel", "shopping_net", "misc_pos", "entertainment"]:
        risco += 8

    estados_baixo_fluxo = ["AK", "WY", "MT", "ND", "SD", "VT"]
    if estado in estados_baixo_fluxo:
        risco += 10

    if parte_dia == "night":
        risco += 10
    elif parte_dia == "evening":
        risco += 5

    risco = max(0, min(risco, 99))
    return risco

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        transaction_data = {
            "month": data.get("month", 1),
            "day": data.get("day", 1),
            "day_of_week": data.get("day_of_week", 1),
            "hour": data.get("hour", 12),
            "category": data.get("category", "misc_pos"),
            "gender": data.get("gender", "M"),
            "state": data.get("state", "FL"),
            "part_of_day": data.get("part_of_day", "afternoon"),
            "amt": float(data.get("amt", 0))
        }

        df = pd.DataFrame([{
            "month": transaction_data["month"],
            "day": transaction_data["day"],
            "day_of_week": transaction_data["day_of_week"],
            "hour": transaction_data["hour"],
            "category_te": mapeamentos["category"].get(transaction_data["category"], 0),
            "gender_te": mapeamentos["gender"].get(transaction_data["gender"], 0),
            "state_te": mapeamentos["state"].get(transaction_data["state"], 0),
            "part_of_day_te": mapeamentos["part_of_day"].get(transaction_data["part_of_day"], 0),
            "amt": transaction_data["amt"]
        }])

        df["amt_scaled"] = scaler.transform(df[["amt"]])
        df = df.drop("amt", axis=1)

        colunas_esperadas = [
            "month", "day", "day_of_week", "hour",
            "category_te", "gender_te", "state_te", "part_of_day_te", "amt_scaled"
        ]
        df = df[colunas_esperadas]

        probability_base = model.predict_proba(df)[0][1]
        probability_final = calcular_risco_amplificado(transaction_data, probability_base)

        if probability_final > 70:
            risk_level = "ALTO"
        elif probability_final > 30:
            risk_level = "MÉDIO"
        else:
            risk_level = "BAIXO"

        prediction = 1 if risk_level == "ALTO" else 0

        return jsonify({
            "status": "success",
            "fraud": bool(prediction),
            "probability_base": round(probability_base * 100, 2),
            "probability_final": round(probability_final, 2),
            "risk_level": risk_level,
            "message": "Transação suspeita!" if prediction else "Transação normal"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)