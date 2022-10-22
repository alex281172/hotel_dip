import json
import random

from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import nltk

# загрузка модулей nltk
nltk.download('punkt')
nltk.download('stopwords')
# морфологический анализатор для русского языка
morph = MorphAnalyzer()

df = pd.read_csv('data.csv')
print(df.head())

df = df[['query']]
df.columns = ['text']
df = df.dropna()
print(df.text)

# подгружаем стоп слова для русского языка
stopwords_ru = stopwords.words("russian")


# нормализация текстов
def normalize_text(text, stop_words=False):
    # токенизация и приведение к нижнему регистру
    text = nltk.word_tokenize(text.lower())
    # лемматизация
    text = [morph.normal_forms(token)[0] for token in text]
    # удаление стоп-слов
    if stop_words:
        text = [token for token in text if token not in stopwords_ru]
    return " ".join(text)


df["normal_text"] = [normalize_text(text, stop_words=True) for text in df.text]
print(df.normal_text)
print('----------------')
print(df.text)
print('++++++++++++++++')

# векторизуем наши запросы
from sklearn.feature_extraction.text import TfidfVectorizer

"""
TfidfVectorizer() работает следующим образов:
1. преобразует запросы с помощью CountVectorizer() - который суммирует one-hot эмбеддинги всех слов запроса
2. трансформирует полученные эмбеддинги, применяя tf*idf
"""
vectorizer = TfidfVectorizer()
text_embeddings = vectorizer.fit_transform(df.text)

print(text_embeddings)

# импорт кластеризаторов из sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans


def cluster_kmeans(num_clusters, embeddings, init="k-means++", random_state=42):
    clustering_model = KMeans(n_clusters=num_clusters, init=init, n_init=100, random_state=random_state)
    clusters = clustering_model.fit_predict(embeddings)
    return clusters


def cluster_miniBatchKMeans(num_clusters, embeddings, init_size=16, batch_size=16, random_state=42):
    clustering_model = MiniBatchKMeans(n_clusters=num_clusters, init_size=init_size, batch_size=batch_size,
                                       random_state=random_state)
    clusters = clustering_model.fit_predict(embeddings)
    return clusters


# число кластеров
num_clusters = 5
kmeans = cluster_kmeans(num_clusters, text_embeddings)
miniBatchKMeans = cluster_miniBatchKMeans(num_clusters, text_embeddings)

print(kmeans)

# импорт библиотек для визулизации кластеров
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_tsne_pca(embeddings, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(embeddings.shape[0]), size=100, replace=False)

    pca = PCA(n_components=2).fit_transform(embeddings[max_items, :].todense())
    tsne = TSNE(perplexity=15).fit_transform(PCA(n_components=20).fit_transform(embeddings[max_items, :].todense()))

    idx = np.random.choice(range(pca.shape[0]), size=100, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i / max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA')

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE')

    f.show()


plot_tsne_pca(text_embeddings, kmeans)
plot_tsne_pca(text_embeddings, miniBatchKMeans)


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    top_keywords = []
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        l = [labels[t] for t in np.argsort(r)[-n_terms:]]
        l.sort()
        print(','.join(l))
        top_keywords.append(','.join(l))
    return top_keywords


# кол-во самых частотных слов
top_words = 1
cluster_names = get_top_keywords(text_embeddings, kmeans, vectorizer.get_feature_names(), top_words)

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(kmeans):
    clustered_sentences[cluster_id].append(df.text[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print(cluster_names[i])
    print(cluster)
    print("")

with open("marketed_dataset.json", "w") as outfile:
    json.dump({cluster_names[i]: cluster for i, cluster in enumerate(clustered_sentences)}, outfile, ensure_ascii=False)

from sklearn.model_selection import train_test_split

# Разделение датасета на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)



