import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

'''
Objetivo: Treinar o modelo de Regressão Logística para detectar as fraudes
A regressão logística vai se utilizar de probabilidade para prever resultados.
'''

pasta_base = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmosPython"
X_train_path = os.path.join(pasta_base, "X_train_balanced.csv")
y_train_path = os.path.join(pasta_base, "y_train_balanced.csv")
X_test_path = os.path.join(pasta_base, "X_test.csv")
y_test_path = os.path.join(pasta_base, "y_test.csv")


X_train = pd.read_csv(X_train_path, sep=',', encoding='latin-1')
y_train = pd.read_csv(y_train_path, sep=',', encoding='latin-1')
X_test = pd.read_csv(X_test_path, sep=',', encoding='latin-1')
y_test = pd.read_csv(y_test_path, sep=',', encoding='latin-1')


if y_train.shape[1] > 1:
    y_train = y_train.iloc[:, 0]
if y_test.shape[1] > 1:
    y_test = y_test.iloc[:, 0] #garante que os testes sejam arryas e não dataFrames


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nTreinamento concluído!")
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))

pasta_destino = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmosPython\modelosDeIa"
model_path = os.path.join(pasta_destino, "modelo1_logistic.pkl")
joblib.dump(model, model_path)