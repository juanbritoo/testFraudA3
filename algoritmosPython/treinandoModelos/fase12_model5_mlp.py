import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib


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
    y_test = y_test.iloc[:, 0]

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

'''
Objetivo: Treinar uma Rede Neural MLP (Multi-Layer Perceptron)
A rede é composta por 2 camadas ocultas, a primeira com 60 neurônios, que irão identificar padrões e a segunda 
com 30, que irá identificar padrões complexos.
'''

model = MLPClassifier(
    hidden_layer_sizes=(60, 30),
    activation='relu',
    solver='adam',
    learning_rate_init=0.002,
    max_iter=400,
    early_stopping=True,               
    n_iter_no_change=15,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nTreinamento concluído!")
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))


pasta_destino = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmosPython\treinandoModelos"
modelo_path = os.path.join(pasta_destino, "modelo5_mlp.pkl")
joblib.dump(model, modelo_path)