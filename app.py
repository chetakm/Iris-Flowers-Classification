from flask import Flask, render_template, request
import utils
import sklearn

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

import pickle
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # print("Hello")
        sepal_length = float(request.form.get('sepal_length'))
        sepal_width = float(request.form.get('sepal_width'))
        petal_length = float(request.form.get('petal_length'))
        petal_width = float(request.form.get('petal_width'))
    # print(sepal_length)
    prediction = utils.preprocessdata(sepal_length, sepal_width, petal_length, petal_width)

    # trained_model = pickle.load(open('model.pkl', 'rb'))
    # prediction = trained_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    # prediction = utils.preprocessdata(sepal_length, sepal_width, petal_length, petal_width)
    # print(prediction)
    return render_template('/predict.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
