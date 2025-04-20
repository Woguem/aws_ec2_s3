import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def train_model(save_path="iris_model.pkl", model_type="random_forest"):
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier()
    if model_type == "random_forest":
        model = RandomForestClassifier()
    elif model_type == "svm":
        model = SVC()
    elif model_type == "logistic_regression":
        model = LogisticRegression()
    else:
        raise ValueError(f"Modèle non supporté: {model_type}")
    model.fit(X, y)
    joblib.dump(model, save_path)
    #print(iris.target_names)
    return model

def load_model(model_path="iris_model.pkl"):
    return joblib.load(model_path)

def predict(model, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    iris = load_iris()
    return iris.target_names[prediction]

#model = train_model()