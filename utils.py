import numpy as np
import joblib
import pickle


def preprocessdata(sepal_length, sepal_width, petal_length, petal_width):
    test_data = [[sepal_length, sepal_width, petal_length, petal_width]]

    # trained_model = pickle.load("model.pkl")
    trained_model = joblib.load(open('model.pkl', 'rb'))
    print(trained_model)
    prediction = trained_model.predict(test_data)
    print(prediction)
    return prediction
