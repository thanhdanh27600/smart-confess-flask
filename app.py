# Load libraries
import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
from tensorflow.python.eager.context import eager_mode

# instantiate flask
app = flask.Flask(__name__)

# we need to redefine our metric function in order
# to use it when loading the model
MODEL_PATH = "../example-2/"

# load the model, and pass in the custom metric function
model = load_model(f'{MODEL_PATH}games.h5')

# define a predict function as an endpoint


@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x = pd.DataFrame.from_dict(params, orient='index').transpose()
        print(x)
        data["prediction"] = "OK"  # str(model.predict(x)[0][0])
        print(model.predict(x))
        data["success"] = True

    # return a response in json format
    return flask.jsonify(data)


# start the flask app, allow remote connections
app.run(host='localhost', debug=True)
