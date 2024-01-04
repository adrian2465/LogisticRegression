from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD, SparsePCA
from sklearn.feature_extraction.text import CountVectorizer
import logging

stop_words = ['the', 'if', 'and', 'but', 'or', 'with', 'for', 'not', 'with', 'by', 'indicating', 'indicates', 'noted']


def generate_model(sentences, sentences_train, sentences_test, y_train, y_test, num_labels, filename):
    logging.info("Generating model using logistic_regression_SVD_PCA_BOW")
    # Vectorize
    x_train, x_test, num_features, vectorizer = vectorize(sentences, sentences_train, sentences_test)

    # Train
    model = train(x_train, y_train, num_features)

    # Evaluate
    count_misclassified, accuracy = evaluate(model, x_test, y_test)
    logging.info('Misclassified samples: {}'.format(count_misclassified))
    logging.info('Accuracy: {:.2f}'.format(accuracy))
    return vectorizer ,model


def vectorize(sentences_all, sentences_train, sentences_test):
    logging.info("Stop words: "+str(stop_words))
    # Prepare for training. Fit the sentences t oa vocabulary/vectorizer....
    logging.info("Training set=" + str(len(sentences_train)) + ", test set="+str(len(sentences_test))+" sentences")
    vectorizer = CountVectorizer(analyzer='word', stop_words=stop_words)
    logging.info("Vectorizing...")
    vectorizer.fit(sentences_all)
    logging.info("Generating bags of words. Vocabulary="+str(len(vectorizer.vocabulary_))+" words")
    x_train = vectorizer.transform(sentences_train)
    x_test = vectorizer.transform(sentences_test)
    num_features = len(vectorizer.vocabulary_)
    return x_train, x_test, num_features, vectorizer


def train(x_train, y_train, num_labels):
    # Use SVD to start latent semantic analysis
    n_components = num_labels
    n_iter = 10
    logging.info("Performing Truncated SVD transformation for "+str(n_components)+" components, "+str(n_iter)+" iterations.")
    svd = TruncatedSVD(n_iter=n_iter, random_state=42)
    x_train_svd = svd.fit_transform(x_train)

    logging.info("Performing Sparse PCA transformation for "+str(n_components)+" components")
    spca = SparsePCA(n_components=n_components, random_state=42)
    x_train_tran = spca.fit_transform(x_train_svd)

    # Fit the model
    model = LogisticRegression(solver='lbfgs', multi_class='auto')
    model.fit(x_train_tran, y_train)
    return model


def evaluate(model, x_test, y_test):
    logging.info("Making predictions ...")
    # use the model to make predictions with the test data
    y_pred = model.predict(x_test)

    logging.info("Calculating accuracy")

    count_misclassified = (y_test != y_pred).sum()

    accuracy = metrics.accuracy_score(y_test, y_pred)
    return count_misclassified, accuracy
