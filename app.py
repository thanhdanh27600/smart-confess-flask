# Load libraries
import os
import pickle
import re
import sys
from os import listdir

from flask import Flask
import gensim
# import gensim
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from underthesea import word_tokenize

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ------------


sep = os.sep  # directory separator
data_folder = "data"  # folder that contains data and model
data_file = "Data_final.csv"
model_version = "final"

# Set this to False if you want to reuse an existing model with model_version
enable_train_new_model = True

PAD_LEN = 500  # The maximum length of a sentence

# Dictionary to scale the dataset for a more balanced dataset
freq = dict({("#tìmngườiyêu", 3), ("#lcd", 3), ("#gópý", 18), ("#bócphốt", 10),
             ("#hỏiđáp", 2), ("#tìmbạn", 2), ("#tâmsự", 1), ("#chiasẻ", 1)})


def loadDataFromCSV():
    df = pd.read_csv(data_folder + sep + data_file)
    df['tag'] = df['tag'].fillna("#LCD")
    return df

# Create a tokenizer to use later on


def txtTokenizer(texts):
    # texts: A list of sentences
    tokenizer = Tokenizer()
    # fit the tokenizer on our text
    tokenizer.fit_on_texts(texts)

    # get all words that the tokenizer knows
    word_index = tokenizer.word_index
    return tokenizer, word_index

# Remove trash symbols and spaces + Lower the case of all data


def preProcess(sentences):
    # Split sentences according to ; * \n . ? !
    text = re.split('; |\*|\n|\.|\?|\!', sentences)

    # Remove the " \ /
    text = [re.sub(r'|,|"|\\|\/', '', sentence) for sentence in text]

    # VNmese compound noun
    text = [word_tokenize(sentence, format="text") for sentence in text]

    # lowercase everything and remove all unnecessary spaces
    text = [sentence.lower().strip().split()
            for sentence in text if sentence != '']
    return text

# Pre-process tags to become lowercase and standardize them into 8 categories


def preProcessTag(tag):
    temp = tag.lower().replace(" ", "")
    if "ngườiyêu" in temp:
        return "#tìmngườiyêu"
    elif "tâmsự" in temp:
        return "#tâmsự"
    elif "gópý" in temp:
        return "#gópý"
    elif "bócphốt" in temp:
        return "#bócphốt"
    elif "hỏiđáp" in temp:
        return "#hỏiđáp"
    elif "bạn" in temp or "info" in temp or "ngườiđichơi" in temp:
        return "#tìmbạn"
    elif "chiasẻ" in temp:
        return "#chiasẻ"
    elif "lcd" in temp:
        return "#lcd"
    else:
        return "error"

# load the data from the dataframe and do pre-processing to the sentences
# in each confessions as well as labelling them


def loadData(df):
    texts = []
    labels = []
    for sample in df['content']:
        sentences = preProcess(sample)
        tag = df.loc[df.content == sample, 'tag'].values[0]
        label = [preProcessTag(tag) for _ in sentences]
        # [tag tag tag tag tag]
        for i in range(freq[preProcessTag(tag)]):
            texts = texts + sentences
            labels = labels + label
    return texts, labels


file = open(data_folder + sep + "tokenizer_" + model_version + ".json")
tokenizer = tokenizer_from_json(file.read())
file = open(data_folder + sep + "data_" + model_version + ".pkl", 'rb')
X, Y, texts = pickle.load(file)
file.close()
# Split train an test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, shuffle=True)
# word_model = gensim.models.Word2Vec.load(data_folder + sep + "word_model_" + model_version + ".save")
model = load_model(data_folder + sep +
                   "predict_model_" + model_version + ".save")

# ----------

app.config['JSON_AS_ASCII'] = False


@app.route("/", methods=["GET"])
def home():
    response = "Hello"

    return response


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    params = flask.request.json

    input_string = params['msg']

    X_dev = tokenizer.texts_to_sequences(preProcess(input_string))
    X_dev = pad_sequences(X_dev, maxlen=PAD_LEN)
    print("Predicting...")
    result_prediction_dict = dict()
    prediction_cus = model.predict(X_dev, verbose=1)
    print(tokenizer.sequences_to_texts(X_dev))
    for i in range(len(prediction_cus)):
        result_tag = Y_train.columns[np.argmax(prediction_cus[i])]
        result_prediction_dict[result_tag] = result_prediction_dict.get(
            result_tag, 0) + 1
    print(result_prediction_dict)
    print(max(zip(result_prediction_dict.values(),
                  result_prediction_dict.keys()))[1])

    data["success"] = True

    data["tags"] = max(zip(result_prediction_dict.values(),
                           result_prediction_dict.keys()))[1]

    # return a response in json format
    return flask.json.dumps(data, ensure_ascii=False)


app.run()
