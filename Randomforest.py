import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 60)

df=pd.read_csv('randomforest.csv')
print(df.head)
df['job'] = pd.factorize(df.job)[0]
df['marital'] = pd.factorize(df.marital)[0]
df['education'] = pd.factorize(df.education)[0]
df['default'] = pd.factorize(df.default)[0]
df['housing'] = pd.factorize(df.housing)[0]
df['month'] = pd.factorize(df.month)[0]
df['poutcome'] = pd.factorize(df.poutcome)[0]
df['y'] = pd.factorize(df.y)[0]
df['contact'] = pd.factorize(df.contact)[0]
df['loan'] = pd.factorize(df.loan)[0]


print(df.head)

X=df.iloc[:, :-1].values
y=df.iloc[:, -1].values


# X=df.values
# y=df['loan'].values

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.8, random_state=0)

sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier=RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

