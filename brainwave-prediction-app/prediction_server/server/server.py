from flask import Flask, render_template, request
import pandas as pd
import json
import tensorflow as tf
import numpy as np
import tensorflow_decision_forests as tfdf


new_model = tf.keras.models.load_model(
    "../will_modelv3")
prediction_cache = []
labels = ['backward', 'down', 'forward',
          'land', 'left', 'right', 'takeoff', 'up']
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/eegrandomforestprediction', methods=['POST'])
def eegprediction():

    # import and process the data to feed to ML model
    data = request.data
    dict = json.loads(data)
    df = pd.DataFrame.from_dict(dict)

    # Convert to Tensorflow Format
    evaldf = tfdf.keras.pd_dataframe_to_tf_dataset(df)

    # Give to model for it to predict
    prediction = new_model.predict(evaldf)

    # Extract prediction label and append to prediction cache
    pred_idex = np.argmax(prediction, axis=1)

    # map prediction to label index
    predicted_label = labels[pred_idex[0]]

    prediction_cache.append(predicted_label)

    return {"prediction_label": predicted_label, "prediction_count": len(prediction_cache)}


@app.route('/lastprediction', methods=['GET'])
def lastprediction():
    try:
        if prediction_cache:
            return {"prediction_label": str(prediction_cache[-1]), "prediction_count": len(prediction_cache)}
    except:
        print("Error reading cache")
        return {"prediction_label": 'flip'}


if __name__ == '__main__':
    app.run(host='0.0.0.0')
