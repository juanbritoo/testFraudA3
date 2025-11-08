import pandas as pd
import os
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

pasta_base = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmosPython"
pasta_modelos = os.path.join(pasta_base, "treinandoModelos")

modelo1 = joblib.load(os.path.join(pasta_modelos, "modelo1_logistic.pkl"))
modelo2 = joblib.load(os.path.join(pasta_modelos, "modelo2_randomForest.pkl"))
modelo3 = joblib.load(os.path.join(pasta_modelos, "modelo3_xgboost.pkl"))
modelo4 = joblib.load(os.path.join(pasta_modelos, "modelo4_catboost.pkl"))
modelo5 = joblib.load(os.path.join(pasta_modelos, "modelo5_mlp.pkl"))

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

ensemble_ponderado = VotingClassifier(
    estimators=[
        ('xgboost', modelo3),
        ('catboost', modelo4),
        ('random_forest', modelo2),
        ('mlp', modelo5),
        ('logistic', modelo1)
    ],
    voting='soft',
    weights=[10, 9, 8, 7, 5],
    n_jobs=-1
)

ensemble_ponderado.fit(X_train, y_train)

y_pred_ensemble = ensemble_ponderado.predict(X_test)
y_prob_ensemble = ensemble_ponderado.predict_proba(X_test)[:, 1]

print("\nTreinamento concluído!")
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_ensemble))
print("AUC:", roc_auc_score(y_test, y_prob_ensemble))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred_ensemble))

ensemble_path = os.path.join(pasta_modelos, "modelo_final_ensemble_ponderado2.pkl")
joblib.dump(ensemble_ponderado, ensemble_path)