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
# X = dataset.loc[:, ["Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Lactic Dehydrogenase", "Albumin", "Hb saturation (arterial blood gases)", "pCO2 (arterial blood gas analysis)"]]
# X = dataset.loc[:, ["Patient age quantile", "Patient addmited to regular ward (1=yes, 0=no)", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes"]]
# X = dataset.loc[:, ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Serum Glucose", "Respiratory Syncytial Virus"]]
# X = dataset.loc[:, ["Influenza A", "Influenza B", "Parainfluenza 1", "CoronavirusNL63", "Rhinovirus/Enterovirus", "Mycoplasma pneumoniae", "Coronavirus HKU1", "Parainfluenza 3", "Chlamydophila pneumoniae", "Adenovirus"]]    # ERRO
# X = dataset.loc[:, ["Parainfluenza 4", "Coronavirus229E", "CoronavirusOC43", "Inf A H1N1 2009", "Bordetella pertussis", "Metapneumovirus", "Parainfluenza 2", "Neutrophils", "Urea", "Proteina C reativa mg/dL"]]
# X = dataset.loc[:, ["Creatinine", "Potassium", "Sodium", "Influenza B, rapid test", "Influenza A, rapid test", "Alanine transaminase", "Aspartate transaminase", "Gamma-glutamyltransferase ", "Total Bilirubin", "Direct Bilirubin"]]
# X = dataset.loc[:, ["Indirect Bilirubin", "Alkaline phosphatase", "Ionized calcium ", "Strepto A", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "Fio2 (venous blood gas analysis)"]]
# X = dataset.loc[:, ["Total CO2 (venous blood gas analysis)", "pH (venous blood gas analysis)", "HCO3 (venous blood gas analysis)", "Rods #", "Segmented", "Promyelocytes", "Metamyelocytes", "Myelocytes", "Myeloblasts", "Urine - Esterase"]]
# X = dataset.loc[:, ["Urine - Aspect", "Urine - pH", "Urine - Hemoglobin", "Urine - Bile pigments", "Urine - Ketone Bodies", "Urine - Nitrite", "Urine - Density", "Urine - Urobilinogen", "Urine - Protein", "Urine - Sugar"]]
# X = dataset.loc[:, ["Urine - Leukocytes", "Urine - Crystals", "Urine - Red blood cells", "Urine - Hyaline cylinders", "Urine - Granular cylinders", "Urine - Yeasts", "Urine - Color", "Partial thromboplastin time (PTT) ", "Relationship (Patient/Normal)", "International normalized ratio (INR)"]]
# X = dataset.loc[:, ["Lactic Dehydrogenase", "Prothrombin time (PT), Activity", "Vitamin B12", "Creatine phosphokinase (CPK) ", "Ferritin", "Arterial Lactic Acid", "Lipase dosage", "D-Dimer", "Albumin", "Hb saturation (arterial blood gases)"]]
# X = dataset.loc[:, ["pCO2 (arterial blood gas analysis)", "Base excess (arterial blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "pO2 (arterial blood gas analysis)", "Arteiral Fio2", "Phosphor", "ctO2 (arterial blood gas analysis)"]]
# X = dataset.loc[:, ["Neutrophils", "Urea", "Proteina C reativa mg/dL", "Creatinine", "Potassium", "Sodium", "Alanine transaminase", "Aspartate transaminase", "Gamma-glutamyltransferase ", "Total Bilirubin"]]
# X = dataset.loc[:, ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Serum Glucose", "pCO2 (arterial blood gas analysis)"]]
# X = dataset.loc[:, ["Patient age quantile", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", "Indirect Bilirubin", "Alkaline phosphatase"]]
# X = dataset.loc[:, ["Ionized calcium ", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "Base excess (arterial blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)"]]
# X = dataset.loc[:, ["pO2 (arterial blood gas analysis)", "ctO2 (arterial blood gas analysis)", "Neutrophils", "Urea", "Proteina C reativa mg/dL", "Creatinine", "Potassium", "Sodium", "Alanine transaminase", "Aspartate transaminase"]]
# X = dataset.loc[:, ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Gamma-glutamyltransferase ", "Total Bilirubin"]]
# X = dataset.loc[:, ["Ionized calcium ", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "Aspartate transaminase"]]
# X = dataset.loc[:, ["ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "Aspartate transaminase"]]
# X = dataset.loc[:, ["Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]]
# X = dataset.loc[:, ["Proteina C reativa mg/dL", "Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]]
# X = dataset.loc[:, ["Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Lactic Dehydrogenase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]]
# X = dataset.loc[:, ["Sodium", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]]
# X = dataset.loc[:, ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]]
# X = dataset.loc[:, ["Proteina C reativa mg/dL", "Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Gamma-glutamyltransferase ", "Total Bilirubin"]]
# X = dataset.loc[:, ["Alanine transaminase", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", "Alkaline phosphatase"]]
# X = dataset.loc[:, ["Sodium", "Alanine transaminase", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes"]]
# X = dataset.loc[:, ["Patient age quantile", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Alanine transaminase", "Aspartate transaminase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]]
X = dataset.loc[:, ["Patient age quantile", "Hematocrit", "Proteina C reativa mg/dL", "Platelets", "Proteina C reativa mg/dL", "Red blood Cells", "Monocytes", "Leukocytes", "Eosinophils", "Hemoglobin"]]

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

# nomes = ["Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Lactic Dehydrogenase", "Albumin", "Hb saturation (arterial blood gases)", "pCO2 (arterial blood gas analysis)"]
# nomes = ["Patient age quantile", "Patient addmited to regular ward (1=yes, 0=no)", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume", "Red blood Cells", "Lymphocytes"]
# nomes = ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Serum Glucose", "Respiratory Syncytial Virus"]
# nomes = ["Influenza A", "Influenza B", "Parainfluenza 1", "CoronavirusNL63", "Rhinovirus/Enterovirus", "Mycoplasma pneumoniae", "Coronavirus HKU1", "Parainfluenza 3", "Chlamydophila pneumoniae", "Adenovirus"]    # ERRO
# nomes = ["Parainfluenza 4", "Coronavirus229E", "CoronavirusOC43", "Inf A H1N1 2009", "Bordetella pertussis", "Metapneumovirus", "Parainfluenza 2", "Neutrophils", "Urea", "Proteina C reativa mg/dL"]
# nomes = ["Creatinine", "Potassium", "Sodium", "Influenza B, rapid test", "Influenza A, rapid test", "Alanine transaminase", "Aspartate transaminase", "Gamma-glutamyltransferase ", "Total Bilirubin", "Direct Bilirubin"]
# nomes = ["Indirect Bilirubin", "Alkaline phosphatase", "Ionized calcium ", "Strepto A", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "Fio2 (venous blood gas analysis)"]
# nomes = ["Total CO2 (venous blood gas analysis)", "pH (venous blood gas analysis)", "HCO3 (venous blood gas analysis)", "Rods #", "Segmented", "Promyelocytes", "Metamyelocytes", "Myelocytes", "Myeloblasts", "Urine - Esterase"]    # ERRO
# nomes = ["Urine - Aspect", "Urine - pH", "Urine - Hemoglobin", "Urine - Bile pigments", "Urine - Ketone Bodies", "Urine - Nitrite", "Urine - Density", "Urine - Urobilinogen", "Urine - Protein", "Urine - Sugar"]    # ERRO
# nomes = ["Urine - Leukocytes", "Urine - Crystals", "Urine - Red blood cells", "Urine - Hyaline cylinders", "Urine - Granular cylinders", "Urine - Yeasts", "Urine - Color", "Partial thromboplastin time (PTT) ", "Relationship (Patient/Normal)", "International normalized ratio (INR)"]    # ERRO
# nomes = ["Lactic Dehydrogenase", "Prothrombin time (PT), Activity", "Vitamin B12", "Creatine phosphokinase (CPK) ", "Ferritin", "Arterial Lactic Acid", "Lipase dosage", "D-Dimer", "Albumin", "Hb saturation (arterial blood gases)"]  # ERRO
# nomes = ["pCO2 (arterial blood gas analysis)", "Base excess (arterial blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "pO2 (arterial blood gas analysis)", "Arteiral Fio2", "Phosphor", "ctO2 (arterial blood gas analysis)"]
# nomes = ["Neutrophils", "Urea", "Proteina C reativa mg/dL", "Creatinine", "Potassium", "Sodium", "Alanine transaminase", "Aspartate transaminase", "Gamma-glutamyltransferase ", "Total Bilirubin"]
# nomes = ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Serum Glucose", "pCO2 (arterial blood gas analysis)"]
# nomes = ["Patient age quantile", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", "Indirect Bilirubin", "Alkaline phosphatase"]
# nomes = ["Ionized calcium ", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "Base excess (arterial blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)"]
# nomes = ["pO2 (arterial blood gas analysis)", "ctO2 (arterial blood gas analysis)", "Neutrophils", "Urea", "Proteina C reativa mg/dL", "Creatinine", "Potassium", "Sodium", "Alanine transaminase", "Aspartate transaminase"]
# nomes = ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Gamma-glutamyltransferase ", "Total Bilirubin"]
# nomes = ["Ionized calcium ", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "Aspartate transaminase"]
# nomes = ["ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "Aspartate transaminase"]
# nomes = ["Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]
# nomes = ["Proteina C reativa mg/dL", "Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]
# nomes = ["Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Lactic Dehydrogenase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]
# nomes = ["Sodium", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]
# nomes = ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]
# nomes = ["Proteina C reativa mg/dL", "Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Gamma-glutamyltransferase ", "Total Bilirubin"]
# nomes = ["Alanine transaminase", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", "Alkaline phosphatase"]
# nomes = ["Sodium", "Alanine transaminase", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes"]
# nomes = ["Patient age quantile", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Alanine transaminase", "Aspartate transaminase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]
nomes = ["Patient age quantile", "Hematocrit", "Proteina C reativa mg/dL", "Platelets", "Proteina C reativa mg/dL", "Red blood Cells", "Monocytes", "Leukocytes", "Eosinophils", "Hemoglobin"]

#Plota gráfico, salvo na mesma pasta do código
shap.summary_plot(shap_values[1], X_train, show=False, max_display=10, feature_names=nomes)
plt.savefig('exam_all_28.png', bbox_inches='tight')