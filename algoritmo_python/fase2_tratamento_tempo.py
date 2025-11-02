import pandas as pd
import os


caminho_arquivo = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmo_python\A3 - Fraud Guard - Tratado.csv"


df = pd.read_csv(caminho_arquivo, encoding="latin-1", sep=';')

'''
Objetivo: Tratar a coluna de tempo
Primeiro utilizamos o datetime do pandas para transformar os dados da coluna trans_date_trans_time, depois
fazemos o tratamento para separar cada dado, sendo eles: mês, dia, dia da semanda (sendo 0 seg e 6 dom), e a hora.
Além de criarmos uma nova coluna com base na hora para saber os períodos do dia.
'''

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')


df['month'] = df['trans_date_trans_time'].dt.month
df['day'] = df['trans_date_trans_time'].dt.day
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek  #0: segunda | 6: domingo
df['hour'] = df['trans_date_trans_time'].dt.hour


def get_part_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 23:
        return 'evening'
    else:
        return 'night'

df['part_of_day'] = df['hour'].apply(get_part_of_day)


df = df.drop(columns=['trans_date_trans_time', 'year', 'is_weekend'], errors='ignore')

novo_caminho = os.path.join(os.path.dirname(caminho_arquivo), "A3 - Fraud Guard - TempoTratado.csv")
df.to_csv(novo_caminho, index=False, encoding="latin-1", sep=';')

print("Tratamento Concluído!")

