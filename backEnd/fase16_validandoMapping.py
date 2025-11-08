import pandas as pd
import joblib
import numpy as np
import os

df_target = pd.read_csv(r"C:\Users\juan_\Downloads\testFraudA3-main\pre_processamento_baseDeDados\A3 - Fraud Guard - TargetEncoding.csv", sep=';', encoding='latin-1')
mapeamentos = joblib.load(r"C:\Users\juan_\Downloads\testFraudA3-main\backEnd\mapeamentos_target_encoding.joblib")

TOLERANCIA = 0.0000001
categorias_todas = list(mapeamentos['category'].keys())
encontradas = 0

for categoria in categorias_todas:
    valor_mapeamento = mapeamentos['category'].get(categoria)
    matches = df_target[np.abs(df_target['category_te'] - valor_mapeamento) < TOLERANCIA]
    if not matches.empty:
        encontradas += 1

print(f"Categorias validadas: {encontradas}/{len(categorias_todas)}")