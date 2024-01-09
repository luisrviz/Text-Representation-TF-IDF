from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing


def representation_binary_weigth(df):
    count_vectorizer = CountVectorizer(analyzer="word", binary=True)
    training_text_features = count_vectorizer.fit_transform(df.clean_only_text)
    return training_text_features


def representation_tf(df):
    count_vectorizer = CountVectorizer(analyzer="word")
    training_text_features = count_vectorizer.fit_transform(df.clean_only_text)
    tfidf = TfidfTransformer(norm="l2", use_idf=False)
    training_text_features = tfidf.fit_transform(training_text_features)
    return training_text_features


def representation_log_tf(df):
    count_vectorizer = CountVectorizer(analyzer="word")
    training_text_features = count_vectorizer.fit_transform(df.clean_only_text)
    tfidf = TfidfTransformer(norm=None, use_idf=False, sublinear_tf=True)
    training_text_features = tfidf.fit_transform(training_text_features)
    return training_text_features


def representation_tf_idf(df):
    count_vectorizer = CountVectorizer(analyzer="word")
    training_text_features = count_vectorizer.fit_transform(df.clean_only_text)
    tfidf = TfidfTransformer(norm="l2")
    training_text_features = tfidf.fit_transform(training_text_features)
    return training_text_features

# Ahora definimos la función que aplica la reducción de dimensionalidad
# Su input es el output de la matriz término documento resultante de aplicar las funciones de pesado

def dimensionality_reduction_lsi(training_text_features, num_components):
    # Create SVD object
    training_text_features.shape[1]
    lsa = TruncatedSVD(n_components=num_components, n_iter=10)
    # Fit SVD model on data
    training_text_features_reduced = lsa.fit_transform(training_text_features)
    training_text_features_reduced = preprocessing.normalize(training_text_features_reduced)
    return training_text_features_reduced
