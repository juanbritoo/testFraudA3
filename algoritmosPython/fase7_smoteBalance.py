import pandas as pd
import os
from imblearn.over_sampling import SMOTE


X_train = pd.read_csv(r'C:\Users\juan_\Downloads\testFraudA3-main\algoritmosPython\X_train.csv')
y_train = pd.read_csv(r'C:\Users\juan_\Downloads\testFraudA3-main\algoritmosPython\Y_train.csv')

'''
Objetivo: Aplicar nos dados de treino para equilibrar as classes is_fraud.
O smote basicamente irá criar dados sintéticos, para aprender melhor a detectar uma transação fraudulenta.
'''

print("Antes do SMOTE:")
print(y_train['is_fraud'].value_counts())


smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


print("Depois do SMOTE:")
print(y_train_res['is_fraud'].value_counts())

pasta_destino = r'C:\Users\juan_\Downloads\testFraudA3-main\algoritmosPython'

X_train_res.to_csv(os.path.join(pasta_destino,'X_train_balanced.csv'), index=False)
y_train_res.to_csv(os.path.join(pasta_destino,'y_train_balanced.csv'), index=False)

print("Balanceamento Concluído")

#Execute no terminal: pip install imbalanced-learn
