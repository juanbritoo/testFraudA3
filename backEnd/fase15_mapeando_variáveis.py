import pandas as pd
import joblib
import os

df = pd.read_csv(r"C:\Users\juan_\Downloads\testFraudA3-main\pre_processamento_baseDeDados\A3 - Fraud Guard - NaNsTratado.csv", sep=';', encoding='latin-1')
df.columns = df.columns.str.strip()

mapeamentos = {}
for col in ['category', 'gender', 'state', 'part_of_day']:
    mean_encoding = df.groupby(col)['is_fraud'].mean()  
    mapeamentos[col] = mean_encoding

pasta_destino = r"C:\Users\juan_\Downloads\testFraudA3-main\backEnd"
os.makedirs(pasta_destino, exist_ok=True)

caminho_arquivo = os.path.join(pasta_destino, 'mapeamentos_target_encoding.joblib')
joblib.dump(mapeamentos, caminho_arquivo)
