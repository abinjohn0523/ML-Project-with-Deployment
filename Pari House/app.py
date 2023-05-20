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
    squareMeters = float(request.form["squareMeters"])
    numPrevOwners = float(request.form["numPrevOwners"])
    numberOfRooms = float(request.form["numberOfRooms"])
    cityPartRange = float(request.form["cityPartRange"])
    hasStormProtector = float(request.form["hasStormProtector"])
    floors = float(request.form["floors"])




    x = np.array([[squareMeters,numPrevOwners,numberOfRooms,cityPartRange,hasStormProtector,floors]])
    x = RobustScaler().fit_transform(x)
    model = pickle.load(open('MODEL.pkl', 'rb'))
    Y_predict = model.predict(x)
    return jsonify({'Prediction': float(Y_predict)})


if __name__ == '__main__':
    app.run(debug=True, port=1010)
