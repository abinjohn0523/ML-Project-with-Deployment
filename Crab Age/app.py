import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
import sklearn
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)


@app.route('/')
def Home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def results():
    Shell_Weight = float(request.form["Shell_Weight"])
    Height = float(request.form["Height"])
    Diameter = float(request.form["Diameter"])
    Length = float(request.form["Length"])
    Weight = float(request.form["Weight"])
    Viscera_Weight = float(request.form["Viscera_Weight"])
    Shucked_Weight = float(request.form["Shucked_Weight"])



    x = np.array([[Shell_Weight,Height,Diameter,Length,Weight,Viscera_Weight,Shucked_Weight]])
    x = RobustScaler().fit_transform(x)
    model = pickle.load(open('MODEL.pkl', 'rb'))
    Y_predict = model.predict(x)
    return jsonify({'Prediction': float(Y_predict)})


if __name__ == '__main__':
    app.run(debug=True, port=1010)
