import pandas as pd
import os  

'''
Primeiro passo é descompactar o arquivo e rodar este código, ele irá pegar a base de dados original
fazer a exclusão das colunas desnecessárias e substituir o arquivo original. Isso para não ter um arquivo 
pesado dentro da pasta.
'''

caminho_original = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmo_python\A3 - Fraud Guard.csv"
caminho_tratado = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmo_python\A3 - Fraud Guard - Tratado.csv"


df = pd.read_csv(caminho_original, encoding="latin-1", sep=None, engine="python")


colunas_para_remover = [
    'Column1', 'cc_num', 'trans_num', 'first', 'last', 'job', 'dob', 'merchant',
    'city', 'zip', 'street', 'lat', 'long', 'city_pop', 'unix_time',
    'merch_lat', 'merch_long'
]


df_tratado = df.drop(columns=colunas_para_remover, errors='ignore')


df_tratado.to_csv(caminho_tratado, index=False, encoding="latin-1", sep=';')


if os.path.exists(caminho_original):
    os.remove(caminho_original)
    print("Arquivo foi removido com sucesso!")
else:
    print("Arquivo não encontrado.")