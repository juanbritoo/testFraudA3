import pandas as pd
import os
from xgboost import XGBClassifier
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


'''
Objetivo: Treinar o eXtreme Gradient Boosting
Baseado nas árvores de decisão, ele treina 200 árvores, e a cada árvore ele corrige 10% do erro da anterior.
'''

if y_train.shape[1] > 1:
    y_train = y_train.iloc[:, 0]
if y_test.shape[1] > 1:
    y_test = y_test.iloc[:, 0]


model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nTreinamento concluido!")
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))

pasta_destino = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmosPython\treinandoModelos"
model_path = os.path.join(pasta_destino, "modelo3_xgboost.pkl")
joblib.dump(model, model_path)

# rode no terminal: pip install xgboost