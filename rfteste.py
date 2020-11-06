import pandas as pd
import numpy as np

dataset = pd.read_excel(r'dataset.xlsx')

dataset = dataset.replace(['not_detected'], float(0))
dataset= dataset.replace(['detected'], float(1))
dataset= dataset.replace(['not_done'], float(-1))
dataset = dataset.replace(['negative'], float(0))
dataset= dataset.replace(['positive'], float(1))

dataset=dataset.fillna(dataset.median ())

# X = dataset.iloc[:, [ 7, 10, 11, 41, 47, 48, 92, 100, 101, 102]].values
# X = dataset.loc[:, ["Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Lactic Dehydrogenase", "Albumin", "Hb saturation (arterial blood gases)", "pCO2 (arterial blood gas analysis)"]]

# X = dataset.iloc[:, [ 8, 10, 11, 41, 47, 48, 101, 102, 105, 109]].values
X = dataset.iloc[:, [ 6, 7, 8, 40, 48, 50, 52, 100, 102, 104]].values
# X = dataset.iloc[:, [ 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]].values
# X = dataset.iloc[:, [ 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]].values
# X = dataset.iloc[:, [ 1, 8, 9, 10, 13, 18, 26, 35, 41, 55]].values
# X = dataset.iloc[:, [ 1, 39, 41, 60, 101]].values

y = dataset.iloc[:, 2].values
# y = dataset.loc[:, "SARS-Cov-2 exam result"]

# 7, 10, 11, 41, 47, 48, 92, 100, 101,102,103,104,105,  106,107,108
# print (y)
# print (X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# # Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='gini', bootstrap=True)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred, normalize=False))

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# SEED = 80
def intervalo_prec(results):
    mean = results.mean()
    print('Precisão média: {:.2f}%'.format(mean*100))


# np.random.seed(SEED)
cv = StratifiedKFold(n_splits = 5, shuffle = True)
el = LogisticRegression(solver='saga')
results = cross_val_score(rf, X_train, y_train, cv = cv, scoring = 'precision_macro') # accuracy scores of different folds
# print("results = ", results)
# print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std() * 2))  # mean accuracy of all folds
intervalo_prec(results)

from sklearn import metrics
# score = rf.score(X_test,y_test)
# print(score)
predict = rf.predict(X_test)
accuracy = accuracy_score(y_test, predict) * 100
print('Acurácia: ', accuracy, '%')

# pscore = metrics.accuracy_score(y_test, pred)
# print (pscore)

import shap
import matplotlib.pyplot as plt

# load JS visualization code to notebook
shap.initjs()

# explain the model's predictions using SHAP values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train, approximate=True)

shap.summary_plot(shap_values[1], X_train, show=False)
plt.title('Test 3');
plt.savefig('shap_graph_test3.png', bbox_inches='tight')

# import scikitplot as skplt

# data = [y_test, y_train]

# # skplt.metrics.plot_roc(data[0], data[1])
# # plt.savefig('roc_curve.png')