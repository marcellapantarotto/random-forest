import pandas as pd
# import matplotlib
import sklearn
from sklearn import svm, preprocessing

df = pd.read_excel(r'dataset.xlsx', index_col=0)
df.head()
# print(df)

df = df.fillna(0)       # filling empty cells with zero

# printing the label results for this column
print(df["SARS-Cov-2 exam result"].unique())

# shuffle rows
df = sklearn.utils.shuffle(df)
print(df)

# feature set
X = df.drop("SARS-Cov-2 exam result", axis=1).values
# X = preprocessing.scale(X) 
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