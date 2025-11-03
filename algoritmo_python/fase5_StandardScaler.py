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


joblib.dump(scaler, os.path.join(os.path.dirname(caminho_arquivo), 'scaler_final.joblib'))


novo_caminho = os.path.join(os.path.dirname(caminho_arquivo), "A3 - Fraud Guard - BasePronta.csv")
df.to_csv(novo_caminho, index=False, encoding='latin-1', sep=';')

print("Escalonamento Concluído")
print(df.head())

#Utilize: pip install scikit-learn > serve para rodar o sklearn no vsCode