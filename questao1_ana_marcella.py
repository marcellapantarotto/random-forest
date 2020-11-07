# Universidade de Brasília
# Departamento de Ciência da Computação
# Introdução à Inteligência Artificial, Turma  A, 1/2020
# Prof. Díbio
# Projeto 3 : Random Forest - Questão 1: 'Predição de diagnóstico positivo ou negativo para COVID-19'
# Ana Luísa Salvador Alvarez - 16/0048036
# Marcella Pantarotto - 13/0143880
# Liguagem utilizada: Python
# Bibliotecas utilizadas: Pandas, Numpy, Scikit Learn, Shap, Matplotlib e as dependências delas


import pandas as pd
import numpy as np

#Importação dos dados obtidos na planilha baixada do keggel
dataset = pd.read_excel(r'dataset.xlsx')

#Substituição de dados não numéricos em representações numéricas
dataset = dataset.replace(['negative'], 0)
dataset= dataset.replace(['positive'], 1)
dataset = dataset.replace(['not_detected'], 0)
dataset= dataset.replace(['detected'], 1)
dataset= dataset.replace(['not_done'], float(-1))

#Preenchimento de valores vazios
dataset=dataset.fillna(dataset.median ())

#Setando valores de Feature set (conjunto de variáveis) e labels
X = dataset.loc[:, ["Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Lactic Dehydrogenase", "Albumin", "Hb saturation (arterial blood gases)", "pCO2 (arterial blood gas analysis)"]]
y = dataset.loc[:, "SARS-Cov-2 exam result"]

#Separando a porção de treino e de teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)

# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Montando a random forest - ajuste da random forest classification  o conjunto de treino
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='gini', bootstrap=True, random_state=200)
rf.fit(X_train, y_train)

# Predição dos resultados do conjunto de teste
y_pred = rf.predict(X_test)

#Relatório de texto, mostrando as principais métricas de classificação
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test,y_pred))

import shap
import matplotlib.pyplot as plt

# Carrega visualização JS
shap.initjs()

# Explioca as predições do modelo usando valores SHAP
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train, approximate=True)

nomes = ["Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Lactic Dehydrogenase", "Albumin", "Hb saturation (arterial blood gases)", "pCO2 (arterial blood gas analysis)"]
#Plota gráfico, salvo na mesma pasta do código
shap.summary_plot(shap_values[1], X_train, show=False, max_display=10, feature_names=nomes)
plt.savefig('shap_graph1.png', bbox_inches='tight')
