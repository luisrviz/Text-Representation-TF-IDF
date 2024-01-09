import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt

# Tal y como dijimos hacemos el análisis para los datos preprocesados, pero antes de la reducción de dimensionalidad
df_preproceso = pd.read_csv("../dataset_preproceso.csv")

# Para determinar la frecuencia de los términos usamos la función que desarrollamos para las visualizaciones, pero
# calculando la frecuencia relativa de esas palabras no la absoluta.

def create_n_gram_frequency(n_gram_from, n_gram_to, corpus):
    vec = CountVectorizer(ngram_range=(n_gram_from, n_gram_to), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    total_num_words = bag_of_words.sum(axis=0).sum()
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, i] / total_num_words) for word, i in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq


def get_frequency_from_words(df_rest, words):
    words_frec = create_n_gram_frequency(1, 1, df_rest)
    words_frec_list = []
    for element in words_frec:
        if element[0] in words:
            words_frec_list.append(element)
    return words_frec_list


def compare_frequencies(df_class, df_rest, num_words):
    particular_class_top_20_frequencies = create_n_gram_frequency(1, 1, df_class)[:num_words]
    words = [unigram[0] for unigram in particular_class_top_20_frequencies]
    general_classes_frequencies = get_frequency_from_words(df_rest, words)
    return particular_class_top_20_frequencies, general_classes_frequencies


def plot(words, particular_class_top_20_frequencies, general_classes_frequencies):
    X_axis = np.arange(len(words))
    plt.bar(X_axis - 0.2, particular_class_top_20_frequencies, 0.4, label='Sport')
    plt.bar(X_axis + 0.2, general_classes_frequencies, 0.4, label='Rest of the classes')

    plt.xticks(X_axis, words)
    plt.xlabel("Words")
    plt.ylabel("Words Frequency")
    plt.title("Word Frequency analysis")
    plt.legend()
    plt.show()


def order_data(df_clase_particular, df_resto_clases):
    particular_class_top_20_frequencies, general_classes_frequencies = \
        compare_frequencies(df_clase_particular,df_resto_clases , 20)
    words = sorted([unigram[0] for unigram in particular_class_top_20_frequencies], key=lambda x: x)
    general_classes_words = [i[0] for i in general_classes_frequencies]
    # We need to consider the case in which words in one topic do not appear in the rest of the topics
    missing_words = [word for word in words if word not in general_classes_words]
    for word in missing_words:
        general_classes_frequencies.append((word, 0))
    # We sort the words alphabetically to make the plots in order
    particular_class_top_20_frequencies = sorted(particular_class_top_20_frequencies, key=lambda x: x[0])
    general_classes_frequencies = sorted(general_classes_frequencies, key=lambda x: x[0])
    # We obtain the frequency alone for the plot
    particular_class_top_20_frequencies = [i[1] for i in particular_class_top_20_frequencies]
    general_classes_frequencies = [i[1] for i in general_classes_frequencies]
    return words, particular_class_top_20_frequencies, general_classes_frequencies


def analysis(words, particular_class_top_20_frequencies, general_classes_frequencies):
    plot(words, particular_class_top_20_frequencies, general_classes_frequencies)


# Realizamos el análisis seleccionando los documentos que queremos comparar (Hay que ver qué posición ocupa cada tipo de
# documentos en nuestro conjunto de datos.
words, particular_class_top_20_frequencies, general_classes_frequencies = \
    order_data(df_preproceso.clean_only_text[900:1000], df_preproceso.clean_only_text[:900])
analysis(words, particular_class_top_20_frequencies, general_classes_frequencies)
