import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

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
Objetivo: Treinar o CatBoost - algoritmo que trata features categóricas 
de forma nativa usando "ordered boosting", uma abordagem única que previne
overfitting de forma inovadora.
'''

if y_train.shape[1] > 1:
    y_train = y_train.iloc[:, 0]
if y_test.shape[1] > 1:
    y_test = y_test.iloc[:, 0]


model = CatBoostClassifier(
    iterations=200,             
    learning_rate=0.1,           
    depth=6,                    
    random_state=42,
    verbose=False,              
    auto_class_weights='Balanced', 
    loss_function='Logloss',     
    eval_metric='AUC',           
    boosting_type='Ordered',     
    bootstrap_type='Bernoulli'  
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nTreinamento concluído!")
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))

pasta_destino = r"C:\Users\juan_\Downloads\testFraudA3-main\algoritmosPython\treinandoModelos"
model_path = os.path.join(pasta_destino, "modelo5_catboost.pkl")
joblib.dump(model, model_path)


#Rode no terminal: pip install catboost