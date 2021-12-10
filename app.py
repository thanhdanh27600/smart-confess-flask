# Load libraries
import flask
import pandas as pd
from numpy import loadtxt
from keras.models import load_model
from tensorflow.python.eager.context import eager_mode

# instantiate flask
app = flask.Flask(__name__)

# we need to redefine our metric function in order
# to use it when loading the model
MODEL_PATH = "models/"

# load the model, and pass in the custom metric function
model = load_model(f'{MODEL_PATH}model.h5')
# summarize model.
model.summary()
# load dataset
dataset = loadtxt(f'{MODEL_PATH}pima-indians-diabetes.csv', delimiter=",")

# define a predict function as an endpoint


@app.route("/", methods=["GET"])
def home():
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    # evaluate the model
    score = model.evaluate(X, Y, verbose=0)

    response = "%s: %.2f%%" % (model.metrics_names[1], score[1]*100)

    return response


@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x = pd.DataFrame.from_dict(
            params, orient='index').transpose().astype(float)
        data["prediction"] = str(model.predict(x)[0])
        data["success"] = True

    # return a response in json format
    return flask.jsonify(data)


# start the flask app, allow remote connections
app.run(host='localhost', debug=True)
