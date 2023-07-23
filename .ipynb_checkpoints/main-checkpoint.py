from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from clean_text import clean_text

app = Flask(__name__)


# fonction de chargement du modele
def load_model():
    # Charger le modele entraine
    model = joblib.load('./models/classification_genginemodel_fabrice_deprez_OPP5_062023.pkl')

    # Charger le vectoriseur TF-IDF
    vectorizer = joblib.load('vectorizer.pkl')

    # Charger le MultiLabelBinarizer
    mlb = joblib.load('mlb.pkl')

    return model, vectorizer, mlb


@app.route('/', methods=['GET'])
def welcome():
    return 'Hello, I work!'


@app.route('/predict_tags', methods=['POST'])
def predict_tags():
    # Charger le modele, le vectoriseur et le MultiLabelBinarizer
    model_path = './models/classification_genginemodel_fabrice_deprez_OPP5_062023.pkl'
    model = joblib.load(model_path)

    vectorizer_path = 'vectorizer.pkl'
    vectorizer = joblib.load(vectorizer_path)

    mlb_path = 'mlb.pkl'
    mlb = joblib.load(mlb_path)

    # Charger le fichier CSV a partir de la requete
    csv_file = request.files['file']

    # Lire le fichier CSV avec pandas
    df = pd.read_csv(csv_file)

    # Extraire les colonnes pertinentes pour la prediction des tags
    body_column = 'Body'
    tags_column = 'Tags'
    bodies = df[body_column].tolist()
    tags = df[tags_column].tolist()

    # Pretraiter le texte du body et le combiner avec les tags
    cleaned_bodies = [clean_text(body) for body in bodies]
    combined_texts = [cleaned_body + ' ' + tag for cleaned_body, tag in zip(cleaned_bodies, tags)]

    # Vectoriser les textes d'entree
    input_vectors = vectorizer.transform(combined_texts)

    # Faire des predictions avec le modele
    predicted_labels = model.predict(input_vectors)

    # Decoder les etiquettes predites en utilisant le MultiLabelBinarizer
    predicted_tags = mlb.inverse_transform(predicted_labels)

    # Creer une liste de dictionnaires pour representer les predictions
    predictions = []
    for body, pred_tags in zip(bodies, predicted_tags):
        prediction = {'Body': body, 'PredictedTags': pred_tags}
        predictions.append(prediction)

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
