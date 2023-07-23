from flask import Flask, request, jsonify
from flask_cors import CORS
from tools import clean_text, tokenize, filtering_nouns, lemmatize, LdaModel, SupervisedModel

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return '''
        <h1>Bienvenue à l'API de prédiction des mots clés</h1>
        <p>Cette API fournit deux méthodes pour prédire les mots clés d'un texte.</p>
        <h2>Utilisation</h2>
        <p>POST /predict_tags : Prédit les mots clés d'un texte donné.</p>
        <p>Exemple :</p>
        <code>
            {
                "text": "votre texte ici"
            }
        </code>
        <p>Retourne les mots clés prédits par les modèles supervisé et non supervisé.</p>
        <p>Exemple de réponse :</p>
        <code>
            {
                "text": "votre texte ici",
                "unsupervised_tags": [...],
                "supervised_tags": [...]
            }
        </code>
    '''

class Input:
    def __init__(self, text):
        self.text = text

@app.route('/predict_tags', methods=['POST'])
def get_prediction():
    data = request.json
    input_data = Input(text=data.get('text'))

    cleaned_text = clean_text(data['text'])
    tokenized_text = tokenize(cleaned_text)
    filtered_noun_text = filtering_nouns(tokenized_text)
    lemmatized_text = lemmatize(filtered_noun_text)
    lda_model = LdaModel()
    unsupervised_pred = lda_model.predict_tags(lemmatized_text)
    supervised_model = SupervisedModel()
    supervised_pred = supervised_model.predict_tags(lemmatized_text)
    text = input_data.text

    return jsonify({"text": text,
                    "unsupervised_tags": unsupervised_pred,
                    "supervised_tags": supervised_pred})

#if __name__ == '__main__':
#    app.run(host='0.0.0.0', debug=True)
