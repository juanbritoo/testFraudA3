import pandas as pd
import os

caminho_arquivo = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmo_python\A3 - Fraud Guard - TempoTratado.csv"
df = pd.read_csv(caminho_arquivo, sep=';', encoding='latin-1')

'''
Objetivo: Garantir que não há nenhuma linha sem dado, e preenchendo com o tratamento de NaNs
'''

#1 Textos String
df['category'].fillna('unknown', inplace=True)
df['gender'].fillna('U', inplace=True)
df['state'].fillna('UNK', inplace=True)
df['part_of_day'].fillna('unknown', inplace=True)

#2 Numéricas
df = df.dropna(subset=['amt', 'is_fraud'])

#3 Tempo
df[['month','day','day_of_week','hour']] = df[['month','day','day_of_week','hour']].fillna(0)

novo_caminho = os.path.join(os.path.dirname(caminho_arquivo), "A3 - Fraud Guard - NaNsTratado.csv")
df.to_csv(novo_caminho, index=False, encoding="latin-1", sep=';')

print("Tratamento de NaNs concluído!")
print(df.isnull().sum())
