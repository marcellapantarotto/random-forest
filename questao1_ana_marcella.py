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
import random

#Importação dos dados obtidos na planilha baixada do keggel
dataset = pd.read_excel(r'dataset.xlsx')

#Substituição de dados não numéricos em representações numéricas
dataset = dataset.replace(['negative'], float(0))
dataset= dataset.replace(['positive'], float(1))
dataset = dataset.replace(['not_detected'], float(0))
dataset= dataset.replace(['detected'], float(1))
dataset= dataset.replace(['not_done'], float(-1))
dataset= dataset.replace(['absent'], float(0))
dataset= dataset.replace(['present'], float(1))
dataset = dataset.replace(['normal'], float(0))
dataset = dataset.replace(['Não Realizado'], float(-1))
dataset = dataset.replace(['<1000'], float(random.uniform(0,999)))


#Preenchimento de valores vazios
dataset=dataset.fillna(dataset.median())
dataset=dataset.fillna(0)

# print(dataset["D-Dimer"].unique())

#Setando valores de Feature set (conjunto de variáveis) e labels
X = dataset.loc[:, ["Patient age quantile", "Patient addmited to regular ward (1=yes, 0=no)", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", 
        "Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Serum Glucose", "Respiratory Syncytial Virus", "Influenza A",
        "Influenza B", "Parainfluenza 1", "CoronavirusNL63", "Rhinovirus/Enterovirus",  "Coronavirus HKU1", "Parainfluenza 3", "Chlamydophila pneumoniae", "Adenovirus", "Parainfluenza 4", "Coronavirus229E", "CoronavirusOC43", "Inf A H1N1 2009", "Bordetella pertussis", "Metapneumovirus", 
        "Parainfluenza 2", "Neutrophils", "Urea", "Proteina C reativa mg/dL", "Creatinine", "Potassium", "Sodium", "Influenza B, rapid test", "Influenza A, rapid test", "Alanine transaminase", "Aspartate transaminase", "Gamma-glutamyltransferase ", "Total Bilirubin", "Direct Bilirubin", 
        "Indirect Bilirubin", "Alkaline phosphatase", "Ionized calcium ", "Strepto A", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "Fio2 (venous blood gas analysis)",
        "Total CO2 (venous blood gas analysis)", "pH (venous blood gas analysis)", "HCO3 (venous blood gas analysis)", "Rods #", "Segmented", "Promyelocytes", "Metamyelocytes", "Myelocytes", "Myeloblasts", "Urine - Esterase", "Urine - pH", "Urine - Yeasts", "Urine - Granular cylinders",
        "Urine - Hyaline cylinders", "Urine - Hemoglobin", "Urine - Bile pigments", "Urine - Ketone Bodies", "Urine - Nitrite", "Urine - Urobilinogen", "Urine - Protein","Urine - Density", "Relationship (Patient/Normal)", "International normalized ratio (INR)", "Urine - Red blood cells",
        "Lactic Dehydrogenase", "Creatine phosphokinase (CPK) ", "Ferritin", "Vitamin B12", "Arterial Lactic Acid", "Lipase dosage", "pCO2 (arterial blood gas analysis)", "Base excess (arterial blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)",
        "HCO3 (arterial blood gas analysis)", "pO2 (arterial blood gas analysis)", "Arteiral Fio2", "Phosphor", "ctO2 (arterial blood gas analysis)", "Urine - Leukocytes", "D-Dimer", "Mycoplasma pneumoniae", "Urine - Sugar", "Partial thromboplastin time (PTT) ", "Prothrombin time (PT), Activity"]]

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

nomes = ["Patient age quantile", "Patient addmited to regular ward (1=yes, 0=no)", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", 
        "Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Serum Glucose", "Respiratory Syncytial Virus", "Influenza A",
        "Influenza B", "Parainfluenza 1", "CoronavirusNL63", "Rhinovirus/Enterovirus",  "Coronavirus HKU1", "Parainfluenza 3", "Chlamydophila pneumoniae", "Adenovirus", "Parainfluenza 4", "Coronavirus229E", "CoronavirusOC43", "Inf A H1N1 2009", "Bordetella pertussis", "Metapneumovirus", 
        "Parainfluenza 2", "Neutrophils", "Urea", "Proteina C reativa mg/dL", "Creatinine", "Potassium", "Sodium", "Influenza B, rapid test", "Influenza A, rapid test", "Alanine transaminase", "Aspartate transaminase", "Gamma-glutamyltransferase ", "Total Bilirubin", "Direct Bilirubin", 
        "Indirect Bilirubin", "Alkaline phosphatase", "Ionized calcium ", "Strepto A", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "Fio2 (venous blood gas analysis)",
        "Total CO2 (venous blood gas analysis)", "pH (venous blood gas analysis)", "HCO3 (venous blood gas analysis)", "Rods #", "Segmented", "Promyelocytes", "Metamyelocytes", "Myelocytes", "Myeloblasts", "Urine - Esterase", "Urine - pH", "Urine - Yeasts", "Urine - Granular cylinders",
        "Urine - Hyaline cylinders", "Urine - Hemoglobin", "Urine - Bile pigments", "Urine - Ketone Bodies", "Urine - Nitrite", "Urine - Urobilinogen", "Urine - Protein","Urine - Density", "Relationship (Patient/Normal)", "International normalized ratio (INR)", "Urine - Red blood cells",
        "Lactic Dehydrogenase", "Creatine phosphokinase (CPK) ", "Ferritin", "Vitamin B12", "Arterial Lactic Acid", "Lipase dosage", "pCO2 (arterial blood gas analysis)", "Base excess (arterial blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)",
        "HCO3 (arterial blood gas analysis)", "pO2 (arterial blood gas analysis)", "Arteiral Fio2", "Phosphor", "ctO2 (arterial blood gas analysis)", "Urine - Leukocytes", "D-Dimer", "Mycoplasma pneumoniae", "Urine - Sugar", "Partial thromboplastin time (PTT) ", "Prothrombin time (PT), Activity"]

#Plota gráfico, salvo na mesma pasta do código
shap.summary_plot(shap_values[1], X_train, show=False, max_display=10, feature_names=nomes)
plt.savefig('melhores_exames_questao1.png', bbox_inches='tight')

