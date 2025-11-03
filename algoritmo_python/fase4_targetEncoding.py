import pandas as pd
import os

caminho_arquivo = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmo_python\A3 - Fraud Guard - NaNsTratado.csv"
df = pd.read_csv(caminho_arquivo, sep=';', encoding='latin-1')

'''
Objetivo: Tornar os dados de texto em números
Iremos utilizar o  Target Encoding para transformas as variáveis de texto em números que representam risco
de fraude, ou seja, o quanto ela esta associada ao is_fraud = 1.
'''

df.columns = df.columns.str.strip()


cols_encode = ['category', 'gender', 'state', 'part_of_day']

#Aplicação da técnica Target Encoding
for col in cols_encode:
    mean_encoding = df.groupby(col)['is_fraud'].mean()  
    df[col + '_te'] = df[col].map(mean_encoding)        


df = df.drop(columns=cols_encode)


novo_caminho = os.path.join(os.path.dirname(caminho_arquivo), "A3 - Fraud Guard - TargetEncoding.csv")
df.to_csv(novo_caminho, index=False, encoding='latin-1', sep=';')

print("Target Encoding realizado com sucesso!")
print(df.head()) #validando se as colunas foram atualizadas
