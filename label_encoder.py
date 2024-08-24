import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
clf = joblib.load('iris_classifier.pkl')

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder.npy',allow_pickle=True)

