from flask import Flask, render_template, request, redirect, url_for
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# Loading the review model and tokenizer
model = load_model('review.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))


def model_predict(review):

    review_token = tokenizer.texts_to_sequences([review])
    review_padded = pad_sequences(review_token, maxlen=100, padding='post')

    review_predict = (model.predict(review_padded) > 0.5).astype('int32')

    return review_predict[0]


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    review = request.form['review']
    if review == '' or review.isspace():
        return render_template('home.html')

    pred = model_predict(review)

    return render_template('prediction.html', pred=pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
