# %% 1
# Package imports
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import nn_block

def text_clf():
    categories = ['alt.atheism', 'talk.religion.misc',
                'comp.graphics', 'sci.space']

    newsgroups_train = fetch_20newsgroups(subset='train',  categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test',  categories=categories)

    num_train = len(newsgroups_train.data)
    num_test = len(newsgroups_test.data)

    # max_features is an important parameter. You should adjust it.
    vectorizer = TfidfVectorizer(max_features=220)

    X = vectorizer.fit_transform(newsgroups_train.data + newsgroups_test.data)
    X_train = X[0:num_train, :]
    X_test = X[num_train:num_train+num_test, :]

    Y_train = newsgroups_train.target
    Y_test = newsgroups_test.target

    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    clf = nn_block.NNClassifier(
        input_features=220, hidden_features=128, output_features=4)

    clf.fit(X_train.toarray(), Y_train, epsilon=0.0005, num_passes=5000)

    Y_predict = clf.predict(X_test.toarray())

    ncorrect = 0
    for dy in (Y_test - Y_predict):
        if 0 == dy:
            ncorrect += 1

    print('Text classification accuracy is {}%'.format(
        round(100.0*ncorrect/len(Y_test))))

if __name__ == "__main__":
    text_clf()