from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import regex as re
import numpy as np
import pickle

app = Flask(__name__)
my_model = load_model('translation_model.h5')


def predict(sentence):
    final =''

    with open('eng_tokenizer.pickle', 'rb') as handle:
        eng_tokenizer = pickle.load(handle)

    with open('fr_tokenizer.pickle', 'rb') as handle:
        fr_tokenizer = pickle.load(handle)
    
    y_id_to_word = {value: key for key, value in fr_tokenizer.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    sentence = eng_tokenizer.texts_to_sequences([sentence])
    
    sentence = pad_sequences(sentence, maxlen=15, padding='post')
    predictions = my_model.predict(sentence)
    final = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])

    return final


@app.route('/', methods=['GET', 'POST'])
def home():
    answer=''
    if request.method == 'POST':

        english_sentence = request.form['fname']
        french_sentence  = predict(english_sentence)
        answer = "French Translation : {} ".format(french_sentence)

    return render_template('index.html', fname=answer)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)