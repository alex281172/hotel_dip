import pickle

from pymorphy2 import MorphAnalyzer
import pandas as pd
import nltk

# загрузка модулей nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from normalize import normalize_text

nltk.download('punkt')
nltk.download('stopwords')
# морфологический анализатор для русского языка
morph = MorphAnalyzer()

# считываем датасет

df = pd.read_csv("data_marketed.csv")
print(df.head(5))

df = df[['text', 'label']]
df.columns = ['text', 'label']
df = df.dropna()
print(df.head(5))

# нормализируем текст

df["normal_text"] = [normalize_text(text, stop_words=True) for text in df.text]

vectorizer = TfidfVectorizer()

from sklearn.model_selection import train_test_split

X = vectorizer.fit_transform(df.normal_text)
y = list(df.label)

# Разделение датасета на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# =======
# KNeighborsClassifier
# =======

# Выбор оптимального числа соседей
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def find_optimal_k(X_train, y_train, max_k):
    iters = range(1, max_k, 1)
    acc = []
    for k in iters:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc.append(knn.score(X_test, y_test))

    f, ax = plt.subplots(1, 1)
    ax.plot(iters, acc, marker='o')
    ax.set_xlabel('number of K')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by K')

    plt.show()


max_k = 10
find_optimal_k(X_train, y_train, max_k)

n_neighbors = 5
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

y_true, y_pred = y_test, knn.predict(X_test)
print(classification_report(y_true, y_pred, zero_division=1))

## -- пример запроса к модели
result_vector = vectorizer.transform(['забронировать отель'])
print('Запрос: ', knn.predict_proba(result_vector))

# =======
# end KNeighborsClassifier
# =======


# =======
# LogisticRegressionCV
# =======

from sklearn.linear_model import LogisticRegressionCV

logreg_clf = LogisticRegressionCV(multi_class="multinomial")
logreg_clf.fit(X_train, y_train)

print(logreg_clf.score(X_test, y_test))

y_pred = logreg_clf.predict(X_test)
print(classification_report(y_true, y_pred, zero_division=1))

## -- пример запроса к модели
result_vector = vectorizer.transform(['забронировать отель'])
print('Запрос: ', logreg_clf.predict_proba(result_vector))

## -- сохранение рзультата в файл

model_filename = "logref_model.sav"
pickle.dump(logreg_clf, open(model_filename, 'wb'))
pickle.dump(vectorizer, open("vectorizer.pk", 'wb'))

print('Model is saved into to disk successfully Using Pickle')
