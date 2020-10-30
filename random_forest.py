from numpy.core.numeric import normalize_axis_tuple
import pandas as pd
# import matplotlib
# import sklearn
# from sklearn import svm, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

df = pd.read_excel(r'dataset.xlsx', index_col=0)

# print(df.shape)         # see how many rows and columns
# print(df.columns)       # see label of columns
# print("\n")
# print(df.isna().sum())    # see empty columns

# columns_list = ["Patient ID", "Patient age quantile", "SARS-Cov-2 exam result", "Patient addmited to regular ward (1=yes, 0=no)", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", "Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Serum Glucose", "Respiratory Syncytial Virus", "Influenza A", "Influenza B", "Parainfluenza 1", "CoronavirusNL63", "Rhinovirus/Enterovirus", "Mycoplasma pneumoniae", "Coronavirus HKU1", "Parainfluenza 3", "Chlamydophila pneumoniae", "Adenovirus", "Parainfluenza 4", "Coronavirus229E", "CoronavirusOC43", "Inf A H1N1 2009", "Bordetella pertussis", "Metapneumovirus", "Parainfluenza 2", "Neutrophils", "Urea", "Proteina C reativa mg/dL", "Creatinine", "Potassium", "Sodium", "Influenza B, rapid test", "Influenza A, rapid test", "Alanine transaminase", "Aspartate transaminase", "Gamma-glutamyltransferase ", "Total Bilirubin", "Direct Bilirubin", "Indirect Bilirubin", "Alkaline phosphatase", "Ionized calcium ", "Strepto A", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "Fio2 (venous blood gas analysis)", "Total CO2 (venous blood gas analysis)", "pH (venous blood gas analysis)", "HCO3 (venous blood gas analysis)", "Rods #", "Segmented", "Promyelocytes", "Metamyelocytes", "Myelocytes", "Myeloblasts", "Urine - Esterase", "Urine - Aspect", "Urine - pH", "Urine - Hemoglobin", "Urine - Bile pigments", "Urine - Ketone Bodies", "Urine - Nitrite", "Urine - Density", "Urine - Urobilinogen", "Urine - Protein", "Urine - Sugar", "Urine - Leukocytes", "Urine - Crystals", "Urine - Red blood cells", "Urine - Hyaline cylinders", "Urine - Granular cylinders", "Urine - Yeasts", "Urine - Color", "Partial thromboplastin time (PTT) ", "Relationship (Patient/Normal)", "International normalized ratio (INR)", "Lactic Dehydrogenase", "Prothrombin time (PT), Activity", "Vitamin B12", "Creatine phosphokinase (CPK) ", "Ferritin", "Arterial Lactic Acid", "Lipase dosage", "D-Dimer", "Albumin", "Hb saturation (arterial blood gases)", "pCO2 (arterial blood gas analysis)", "Base excess (arterial blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "pO2 (arterial blood gas analysis)", "Arteiral Fio2", "Phosphor", "ctO2 (arterial blood gas analysis)"]

covid_dictionary = {"positive": 1, "negative": 0}   # dictionary for string values in cells
df["SARS-Cov-2 exam result"] = df["SARS-Cov-2 exam result"].map(covid_dictionary)       # replacing string values
# df["Strepto A"] = df["Strepto A"].map(covid_dictionary)       # replacing string values

# # df[df.columns] = df[df.columns].apply(pd.to_numeric, errors='coerce')
df.fillna(df.median(), inplace=True)       # filling empty cells with median value

df = df.loc[:, ["SARS-Cov-2 exam result", "Hemoglobin", "Leukocytes", "Urine - Aspect", "Arterial Lactic Acid", "pH (arterial blood gas analysis)"]]
print(df.head())        # see first 5 rows

# print("\n Urine - Aspect: ", df["Urine - Aspect"].unique())
df["Urine - Aspect"].fillna("-1", inplace=True)       # filling empty cells with median value

X = df.loc[:, ["Leukocytes"]]
y = df["SARS-Cov-2 exam result"]

# X = df.drop("SARS-Cov-2 exam result", axis='columns')
# y = df["SARS-Cov-2 exam result"]

print("\nX = ", X.shape)
print("y = ", y.shape)

logreg = LogisticRegression(solver='lbfgs')

accuracy = cross_val_score(logreg, X, y, cv=5, scoring='accuracy').mean()       # basic cross validated model
print("accuracy = ", accuracy)
# f = open("out_columens_unique.txt", "w")

# null_accuracy = y.value_counts(normalize=True)
# print("\nnull_accuracy:\n", null_accuracy)

# ohe = OneHotEncoder(sparse=False)
# print(ohe.fit_transform(df[["Urine - Aspect"]]))
# print(ohe.categories_)

# column_trans = make_column_transformer((OneHotEncoder(), ["Urine - Aspect"]), remainder='passthrough')
# column_trans.fit_transform(X)
# logreg = LogisticRegression(solver='lbfgs')

# pipe = make_pipeline(column_trans, logreg)
# accuracy2 = cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()        # cross validation the whole pipeline
# print(accuracy2)

# pipe.fit(X, y)

# # checking if there are any empyt cells left
# for i in df:
#     for j in columns_list:
#         f.write(j)
#         f.write(" = ")
#         f.write(str(df[i].unique()))
#         f.write("\n")
#         # print(j, " = ", df[i].unique())
#         # print(df[i].isnull().any(), " = ", j)
# f.close()

# for i in df:
#     print(df[i].unique())


# for x in df["Urine - Color"]:
#     print(x)

# print(df["SARS-Cov-2 exam result"].unique())    # printing the label results for this column

# shuffle rows
# df = sklearn.utils.shuffle(df)
# print(df)

# # feature set
# X = df.drop("SARS-Cov-2 exam result", axis=1).values
# # X = preprocessing.scale(X)        # bring data into one scale
# print("\n X = \n", X)

# # labels
# y = df['SARS-Cov-2 exam result'].values
# print("\n y = \n", y)


###############################################

# test_size = 200

# X_train= X[:-test_size]
# y_train= y[:-test_size]

# X_test= X[:-test_size]
# y_test= y[:-test_size]

# clf = svm.SVR(kernel="linear")
# clf.fit(X_train, y_train)
# clf.score(X_test, y_test)