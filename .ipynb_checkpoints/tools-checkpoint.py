from bs4 import BeautifulSoup
import regex as re
import nltk
from nltk.corpus import stopwords

def clean_text(text):
    # Supprimer les balises HTML
    text = BeautifulSoup(text, 'html.parser').get_text()

    # Supprimer la ponctuation et les caractères spéciaux
    text = re.sub('[^a-zA-Z0-9]', ' ', text)

    # Mettre en minuscule
    text = text.lower()

    # Remplacer les mots spécifiques
    text = re.sub(r'\bC\+\+\b', 'cplusplus', text)
    #attention a casesentivie
    text = re.sub(r'\b5G\b', 'fiveg', text, flags=re.IGNORECASE)
    text = re.sub(r'\b4G\b', 'fourg', text, flags=re.IGNORECASE)
    # Ajoutez d'autres remplacements si nécessaire pour les mots spécifiques

    # Supprimer les mots vides
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatisation
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Rejoindre les tokens en une chaîne de caractères
    cleaned_text = ' '.join(tokens)

    return cleaned_text