import pandas as pd
from sklearn.model_selection import train_test_split
import os


df = pd.read_csv(r'C:\Users\juan_\Downloads\testFraudA3-main\pre_processamento_baseDeDados\A3 - Fraud Guard - BasePronta.csv',
                  sep=None, engine='python')


'''
Objetivo: Separar a base tratada em x_train e x_test = variáveis independentes e 
y_train e y_test = variável target e Salvando tudo em arquivos separados para melhor tratamento e orquestração
dos modelos de IA.
'''
if 'is_fraud' in df.columns:
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
else:
    raise ValueError("Coluna 'is_fraud' não encontrada.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pasta_destino = r'C:\Users\juan_\Downloads\testFraudA3-main\algoritmosPython'

X_train.to_csv(os.path.join(pasta_destino, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(pasta_destino, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(pasta_destino, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(pasta_destino, 'y_test.csv'), index=False)

print("Divisão de treino e teste concluído")
