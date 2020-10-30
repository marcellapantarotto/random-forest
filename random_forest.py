import pandas as pd
# import matplotlib
import sklearn
from sklearn import svm, preprocessing

df = pd.read_excel(r'dataset.xlsx', index_col=0)
df.head()
print(df)

# exame_dictionary = {"detected": 1, "not_detected": 0}   # dictionary for string values in cells
# df = df.map(exame_dictionary)       # replacing string values
# df.head()

# df[df.columns] = df[df.columns].apply(pd.to_numeric, errors='coerce')
df.fillna(df.median(), inplace=True)       # filling empty cells with median value
# print(df.isnull().any())         # checking if there are any empyt cells left

# print(df["SARS-Cov-2 exam result"].unique())    # printing the label results for this column

# shuffle rows
df = sklearn.utils.shuffle(df)
print(df)

# feature set
X = df.drop("SARS-Cov-2 exam result", axis=1).values
# X = preprocessing.scale(X)        # bring data into one scale
print("\n X = \n", X)

# labels
y = df['SARS-Cov-2 exam result'].values
print("\n y = \n", y)


###############################################

# test_size = 100

# X_train= X[:-test_size]
# y_train= y[:-test_size]

# X_test= X[:-test_size]
# y_test= y[:-test_size]

# clf = svm.SVR(kernel="linear")
# clf.fit(X_train, y_train)
# clf.score(X_test, y_test)