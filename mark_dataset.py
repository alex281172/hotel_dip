from pymorphy2 import MorphAnalyzer
import numpy as np
import pandas as pd
import nltk

from normalize import normalize_text

nltk.download('punkt')
from nltk.corpus import stopwords
# морфологический анализатор для русского языка
morph = MorphAnalyzer()
stopwords_ru = stopwords.words("russian")


# считываем датасет

df = pd.read_csv('data.csv')
print('-------------------df.head()--------------------------')
print(df.head())

df = df[['query']]
df.columns = ['text']
df = df.dropna()
print('-------------------df.text--------------------------')
print(df.text)

# нормализируем текст

df["normal_text"] = [normalize_text(text, stop_words=True) for text in df.text]
print('-------------------df.normal_text--------------------------')
print(df.normal_text)
print('-------------------df.head()--------------------------')
print(df.head())

# считаем векторное сходство фраз

from sklearn.feature_extraction.text import TfidfVectorizer

"""
TfidfVectorizer() работает следующим образов:
1. преобразует запросы с помощью CountVectorizer() - который суммирует one-hot эмбеддинги всех слов запроса
2. трансформирует полученные эмбеддинги, применяя tf*idf
"""
vectorizer = TfidfVectorizer()
text_embeddings_my = vectorizer.fit_transform(df.text)
text_embeddings = vectorizer.fit_transform(df.normal_text)
print('-------------------text_embeddings_my--------------------------')
print(text_embeddings_my)
print('-------------------text_embeddings--------------------------')
print(text_embeddings)
print('-------------------****************--------------------------')

# кластеризируем данные

# KMeans

# MiniBatchKMeans

from sklearn.cluster import KMeans, MiniBatchKMeans


def cluster_kmeans(num_clusters, embeddings, init='k-means++', random_state=42):
    clustering_model = KMeans(n_clusters=num_clusters, init=init, n_init=100, random_state=random_state)
    clusters = clustering_model.fit_predict(embeddings)
    return clusters


def cluster_miniBatchKMeans(num_clusters, embeddings, init_size=16, batch_size=16, random_state=42):
    clustering_model = MiniBatchKMeans(n_clusters=num_clusters, init_size=init_size, batch_size=batch_size,
                                       random_state=random_state)
    clusters = clustering_model.fit_predict(embeddings)
    return clusters


num_clusters = 7

kmeans = cluster_kmeans(num_clusters, text_embeddings)

miniBatchKMeans = cluster_miniBatchKMeans(num_clusters, text_embeddings)

print(kmeans)

# визуализация

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

    plt.show()


plot_tsne_pca(text_embeddings, kmeans)
plot_tsne_pca(text_embeddings, miniBatchKMeans)


# находим самые частые слова в каждом кластере

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


top_words = 1
cluster_names = get_top_keywords(text_embeddings, kmeans, vectorizer.get_feature_names(), top_words)
print(cluster_names)

# собираем кластеры в удобный нам вид

clustered_sentences = [[] for i in range(num_clusters)]

df["label"] = ["" for _ in range(len(df.text))]

for sentence_id, cluster_id in enumerate(kmeans):
    clustered_sentences[cluster_id].append(df.text[sentence_id])
    df.label[sentence_id] = cluster_names[cluster_id]

print(df.head())

# выводим их

for i in range(len(clustered_sentences)):
    print(cluster_names[i])
    print(clustered_sentences[i])

df.to_csv("data_marketed.csv")

