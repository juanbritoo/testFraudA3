import pandas as pd


df = pd.read_csv(r"C:\Users\juan_\Downloads\testFraudA3-main\testFraudA3-main\algoritmo_python\A3 - Fraud Guard.csv",
                 encoding="latin-1", sep=None, engine="python")


colunas_para_remover = [
    'Column1', 'cc_num', 'trans_num', 'first', 'last', 'job', 'dob', 'merchant',
    'city', 'zip', 'street', 'lat', 'long', 'city_pop', 'unix_time',
    'merch_lat', 'merch_long'
]

df_tratado = df.drop(columns=colunas_para_remover, errors='ignore')


df_tratado.to_csv(r"C:\Users\juan_\Downloads\testFraudA3-main\testFraudA3-main\algoritmo_python\A3 - Fraud Guard - Tratado.csv",
                  index=False, encoding="latin-1", sep=';')

print("Arquivo tratado!")
