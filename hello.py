from flask import Flask, request, render_template
# import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
app = Flask(__name__)

# filename = 'model_canada.sav'
# loaded_model = joblib.load(filename)
print("Loading Model")
loaded_model = tf.keras.models.load_model("sign.h5")
loaded_model.build()
print("Model Loaded")
# loaded_model.summary()

# loading test data
test_path = './sign_mnist_test.csv'
test = pd.read_csv(test_path)
test_label = test['label']
test = test.drop(['label'], axis=1)
test = test / 255
test = np.array(test)
test = test.reshape(test.shape[0], 28, 28, 1)

alpha = {"0": "a", "1": "b", "2": "c", "3": "d", "4": "e", "5": "f", "6": "g", "7": "h", "8": "i", "9": "j", "10": "k", "11": "l", "12": "m",
         "13": "n", "14": "o", "15": "p", "16": "q", "17": "r", "18": "s", "19": "t", "20": "u", "21": "v", "22": "w", "23": "y", "24": "z"}


@app.route("/")
def hello_world():
    return render_template('form.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        form_data = request.form.get("test")
        predictions = loaded_model.predict(test)
        predictions = np.argmax(predictions, axis=1)
        result_64 = predictions[int(form_data)]
        result = result_64.item()
        print(result, type(result))
        return render_template('result.html', data=alpha[str(result)])
    else:
        return render_template('form.html')
    # return render_template('form.html')


app.run(host='localhost', port=5000)
