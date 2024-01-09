import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


def process_text(text):
    # Eliminamos las expresiones de html
    text = re.sub("(<.*?>)", " ", text)

    # Eliminamos la etiqueta nbsp de html
    text = re.sub("(&nbsp)", " ", text)

    # Eliminamos la palabra html que aparece al principio de todos los documentos
    text = text[4:]

    # Eliminamos urls
    text = text.replace("http", "https")
    text = re.sub("https://\S+|www\.\S+", " ", text)

    # Eliminamos signos de puntuación
    text = re.sub("[^\w\s]", "", text)

    # Consideramos únicamente letras
    text = re.sub("[^a-zA-Z]", " ", text)

    # Eliminamos los espacios en blanco
    text = text.strip()

    # Convertimos todo a minúsculas
    words = text.lower().split()

    #Lematización
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized = [wordnet_lemmatizer.lemmatize(word) for word in words]

    # Eliminamos las stopwords
    stops = set(stopwords.words("english"))
    not_stop_words = [w for w in lemmatized if not w in stops]

    return (" ".join(not_stop_words))


def clean_text(df):
    df["clean_only_text"] = df["only_text"].apply(lambda x: process_text(x))
    return df
