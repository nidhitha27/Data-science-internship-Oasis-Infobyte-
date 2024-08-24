
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv('dataset\Iris.csv',index_col='Id')


X = df.drop('Species', axis=1)
y = df['Species'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

clf=RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train_encoded)
y_pred = clf.predict(X_test)

joblib.dump(clf, 'iris_classifier.pkl')

y = y.unique()
label_encoder = LabelEncoder()
label_encoder.fit(y)
np.save('label_encoder.npy', label_encoder.classes_)