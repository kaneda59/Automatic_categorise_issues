import numpy as np
import pandas as pd
import re
import pickle
from bs4 import BeautifulSoup
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

## Nettoyage
def clean_text(text):
    """
    cette fonction permet de nettoyer un texte, la base du texte est supposée en anglais
    Args:
        text(String): text à nettoyer
    Returns
        cleaned_text(String): texte nettoyé (balise html, majuscule, code spécifique, articles, etc).
    """
    text = BeautifulSoup(text, "html5lib")

    for sent in text(['style', 'script']):
            sent.decompose()

    text = ' '.join(text.stripped_strings)

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


def tokenize(text):
    """
    Tokeniser les mots d'un texte.
    Args:
        text(String): text d'origine
    Returns
        res(list): Chaîne tokenisée.
    """
    stop_words = set(stopwords.words('english'))

    try:
        res = word_tokenize(text, language='english')
    except TypeError:
        return text

    res = [token for token in res if token not in stop_words]
    return res

def filtering_nouns(tokens):
    """
    Filter singular nouns
    Args:
        tokens(list): A list o tokens
    Returns:
        res(list): Filtered token list
    """
    res = nltk.pos_tag(tokens)

    res = [token[0] for token in res if token[1] == 'NN']

    return res

def lemmatize(tokens):
    """
    Transform tokens into lems

    Args:
        tokens(list): List of tokens
    Returns:
        lemmatized(list): List of lemmatized tokens
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized = []

    for token in tokens:
        lemmatized.append(lemmatizer.lemmatize(token))

    return lemmatized

class SupervisedModel:

    def __init__(self):
        filename_supervised_model = "/home/kaneda/api/models/svm_model.pkl"
        filename_mlb_model = "/home/kaneda/api/models/mlb_model.pkl"
        filename_tfidf_model = "/home/kaneda/api/models/tfidf_model.pkl"
        filename_pca_model = "/home/kaneda/api/models/pca_model.pkl"
        filename_vocabulary = "/home/kaneda/api/models/vocabulary.pkl"

        self.supervised_model = pickle.load(open(filename_supervised_model, 'rb'))
        self.mlb_model = pickle.load(open(filename_mlb_model, 'rb'))
        self.tfidf_model = pickle.load(open(filename_tfidf_model, 'rb'))
        self.pca_model = pickle.load(open(filename_pca_model, 'rb'))
        self.vocabulary = pickle.load(open(filename_vocabulary, 'rb'))

    def predict_tags(self, text):
        """
        Predict tags according to a lemmatized text using a supervied model.
        Args:
            supervised_model(): Used mode to get prediction
            mlb_model(): Used model to detransform
        Returns:
            res(list): List of predicted tags
        """
        input_vector = self.tfidf_model.transform(text)
        input_vector = pd.DataFrame(input_vector.toarray(), columns=self.vocabulary)
        input_vector = self.pca_model.transform(input_vector)
        res = self.supervised_model.predict(input_vector)
        res = self.mlb_model.inverse_transform(res)
        res = list({tag for tag_list in res for tag in tag_list if (len(tag_list) != 0)})
        res = [tag for tag  in res if tag in text]

        return res

class LdaModel:

    def __init__(self):
        filename_model = "/home/kaneda/api/models/lda_model.pkl"
        filename_dictionary = "/home/kaneda/api/models/dictionary.pkl"
        self.model = pickle.load(open(filename_model, 'rb'))
        self.dictionary = pickle.load(open(filename_dictionary, 'rb'))

    def predict_tags(self, text):
        """
        Predict tags of a preprocessed text
        Args:
            text(list): preprocessed text
        Returns:
            res(list): list of tags
        """
        corpus_new = self.dictionary.doc2bow(text)
        topics = self.model.get_document_topics(corpus_new)

        #find most relevant topic according to probability
        relevant_topic = topics[0][0]
        relevant_topic_prob = topics[0][1]

        for i in range(len(topics)):
            if topics[i][1] > relevant_topic_prob:
                relevant_topic = topics[i][0]
                relevant_topic_prob = topics[i][1]

        #retrieve associated to topic tags present in submited text
        res = self.model.get_topic_terms(topicid=relevant_topic, topn=20)

        res = [self.dictionary[tag[0]] for tag in res if self.dictionary[tag[0]] in text]

        return res

class NMFModel:

    def __init__(self):
        filename_nmf_model = "/home/kaneda/api/models/nmf_model.pkl"
        filename_vectorizer_model = "/home/kaneda/api/models/vectorizer.pkl"

        self.nmf_model = pickle.load(open(filename_nmf_model, 'rb'))
        self.vectorizer = pickle.load(open(filename_vectorizer_model, 'rb'))

    def predict_tags(self, text):
        """
        Predict tags of a preprocessed text
        Args:
            text(list): preprocessed text
        Returns:
            res(list): list of tags
        """
        text_str = ' '.join(text)  # Convert list of tokens to a single string
        tfidf = self.vectorizer.transform([text_str])  # Pass the string to the transform method
        W = self.nmf_model.transform(tfidf)
        H = self.nmf_model.components_

        # Get the top 10 words for this topic
        feature_names = self.vectorizer.get_feature_names_out()
        top_indices = np.argsort(H[:, np.argmax(W)])[::-1]
        top_feature_names = [feature_names[i] for i in top_indices[:10]]

        # retrieve associated to topic tags present in submitted text
        res = [tag for tag in top_feature_names if tag in text]

        return res

