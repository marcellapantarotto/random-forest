import pandas as pd
import numpy as np

dataset = pd.read_excel(r'dataset.xlsx')

# dataset = dataset.replace(['not_detected'], 0)
# dataset= dataset.replace(['detected'], 1)
dataset = dataset.replace(['negative'], float(0))
dataset= dataset.replace(['positive'], float(1))

dataset=dataset.fillna(dataset.median ())

# X = dataset.iloc[:, [ 7, 10, 11, 41, 47, 48, 92, 100, 101, 102]].values
# X = dataset.iloc[:, [ 8, 10, 11, 41, 47, 48, 101, 102, 105, 109]].values
# X = dataset.iloc[:, [ 6, 7, 8, 40, 48, 50, 52, 100, 102, 104, 109]].values
X = dataset.iloc[:, [ 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]].values
# X = dataset.iloc[:, [ 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]].values
y = dataset.iloc[:, 2].values

# 7, 10, 11, 41, 47, 48, 92, 100, 101,102,103,104,105,  106,107,108
# print (y)
# print (X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# # Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='gini', bootstrap=True, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
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

# print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std() * 2))

# pscore = metrics.accuracy_score(y_test, pred)
# print (pscore)

import shap
import matplotlib.pyplot as plt

# load JS visualization code to notebook
shap.initjs()

# explain the model's predictions using SHAP values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train, approximate=True)

shap.summary_plot(shap_values[1], X_train, show=False, max_display=10)
plt.savefig('shap_graph.png', bbox_inches='tight')

#concatena os valores das colunas referentes a tratamento em enfermaria, na unidade semi-intensiva e na unidade intensiva e armazena em outro dataframe
df1 = dataset['Patient addmited to regular ward (1=yes, 0=no)'].map(str) + dataset['Patient addmited to semi-intensive unit (1=yes, 0=no)'].map(str) + dataset['Patient addmited to intensive care unit (1=yes, 0=no)'].map(str)
print (df1)
#Para considerar tratamento em enfermaria, dado é 100
#Para considerar na semi-intensiva, dado é 010
#Para considerar na uti, dado é 001
#Portanto, conclui-se que os pacientes que se trataram em casa são os de dado 000

#concatena o dataframe df1 com o original, ficando os dados de local e tratamento na coluna 112
dataset = pd.concat([dataset, df1], axis=1)
print (dataset)

#Para a segunda questão, o y passa a ser a coluna nova que foi acionada acima
