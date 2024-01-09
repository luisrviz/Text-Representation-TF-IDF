from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Tal y como dijimos realizamos las visualizaciones sobre el dataset procesado.
df = pd.read_csv("../dataset_preproceso.csv")


def visualize_text_wordcloud(text):
    wordcloud = WordCloud(background_color="black", width=1200, height=800, stopwords=[""]) \
        .generate(' '.join(text.tolist()))
    plt.figure(figsize=(20, 10), facecolor='k')
    plt.imshow(wordcloud)


# La generación de las imágenes puede dar algún problema si se realiza a la vez en función del entorno en que ejecute
def create_n_gram_frequency(n_gram_from, n_gram_to, corpus):
    vec = CountVectorizer(ngram_range=(n_gram_from, n_gram_to), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, i]) for word, i in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq


def visualize_text_top_words(text):
    top_20_unigrams = create_n_gram_frequency(1, 1, text)[:20]
    figure = plt.bar(x=[unigram[0] for unigram in top_20_unigrams],
                    height=[freq[1] for freq in top_20_unigrams])
    plt.xticks(rotation="vertical")
    plt.ylabel('Número de veces que aparece cada unigrama')
    plt.xlabel('unigramas')
    plt.title('Unigramas más comunes')
    plt.show()


# Visualizamos las palabras de los textos sobre Commercial Banks
visualize_text_wordcloud(df.clean_only_text[:100])

# Visualizamos las palabras de los textos sobre Commercial Banks
visualize_text_top_words(df.clean_only_text[:100])
