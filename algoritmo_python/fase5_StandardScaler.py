import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


caminho_arquivo = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmo_python\A3 - Fraud Guard - TargetEncoding.csv"


df = pd.read_csv(caminho_arquivo, sep=';', encoding='latin-1')

'''
Objetivo: Fazer com que os dados da coluna amt tenham o mesmo peso independente do seu valor.
'''

scaler = StandardScaler()

#Aplicando o Standard Scaler na coluna AMT
df['amt_scaled'] = scaler.fit_transform(df[['amt']])


df = df.drop(columns=['amt'])
df = df.rename(columns={'amt_scaled': 'amt'})

novo_caminho = r"C:\Users\juan_\Downloads\testFraudA3-main"
os.makedirs(novo_caminho, exist_ok=True)  

caminho_scaler = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmo_python\scaler_final.joblib"
joblib.dump(scaler, caminho_scaler)

print("Escalonamento Concluído")

#Utilize: pip install scikit-learn > serve para rodar o sklearn no vsCode