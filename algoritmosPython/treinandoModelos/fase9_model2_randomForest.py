import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings

warnings.filterwarnings("ignore")


pasta_base = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmosPython"


X_train_path = os.path.join(pasta_base, "X_train_balanced.csv")
y_train_path = os.path.join(pasta_base, "y_train_balanced.csv")
X_test_path = os.path.join(pasta_base, "X_test.csv")
y_test_path = os.path.join(pasta_base, "y_test.csv")


X_train = pd.read_csv(X_train_path, sep=',', encoding='latin-1')
y_train = pd.read_csv(y_train_path, sep=',', encoding='latin-1')
X_test = pd.read_csv(X_test_path, sep=',', encoding='latin-1')
y_test = pd.read_csv(y_test_path, sep=',', encoding='latin-1')


'''
Objetivo: Treinar o Modelo de Random Forest para a tomada de decisão de fraudes
O funcionamento basicamente funciona criando 200 árvores, com nível de profundidade 10, ou seja faz 10 'perguntas'
sobre a transação e divisão de 10 (10 trasações a se analisar), para evitar overfiting.
'''

if y_train.shape[1] > 1:
    y_train = y_train.iloc[:, 0]
if y_test.shape[1] > 1:
    y_test = y_test.iloc[:, 0]


model = RandomForestClassifier(
    n_estimators=200,        #número de árvores
    max_depth=10,            #profundidade máxima das árvores
    min_samples_split=10,    
    random_state=42,
    n_jobs=-1             
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nTreinamento concluído!")


print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))

pasta_destino = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmosPython\treinandoModelos"
model_path = os.path.join(pasta_destino, "modelo2_randomForest.pkl")
joblib.dump(model, model_path)