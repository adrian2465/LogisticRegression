from sklearn import metrics
from sklearn.svm import SVR
from sklearn.feature_extraction.text import CountVectorizer
import logging
import os.path
import csv

stop_words = ['the', 'if', 'and', 'but', 'or', 'with', 'for', 'not', 'with', 'by', 'indicating', 'indicates', 'noted']


def generate_model(sentences, sentences_train, sentences_test, y_train, y_test, num_labels, filename):
    algorithm = "Logistic Regression Bag of Words"
    num_sentences = len(sentences)
    logging.info("Generating model using logistic_regression_BOW")
    # Vectorize
    x_train, x_test, num_features = vectorize(sentences, sentences_train, sentences_test)

    # Train
    model = train(x_train, y_train)

    # Evaluate
    count_misclassified, accuracy = evaluate(model, x_test, y_test)
    logging.info('Misclassified samples: {}'.format(count_misclassified))
    logging.info('Accuracy: {:.2f}'.format(accuracy))

    # Store results in a CSV.
    fname = "BOW.csv"

    header = ['Algorithm', 'Data file name', 'Data Size', 'Training size', 'Test size', 'Num features', 'Top N labels', 'Misclassified', 'Accuracy']

    if not os.path.isfile(fname):
        with open(fname, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(fname,'a') as f:
        writer = csv.writer(f)
        writer.writerow([algorithm,
                         filename,
                         num_sentences,
                         len(sentences_train),
                         len(sentences_test),
                         num_features,
                         num_labels,
                         count_misclassified,
                         accuracy])

    return model


def vectorize(sentences_all, sentences_train, sentences_test):
    logging.info("Stop words: "+str(stop_words))
    # Prepare for training. Fit the sentences to a vocabulary/vectorizer....
    logging.info("Training set=" + str(len(sentences_train)) + ", test set="+str(len(sentences_test))+" sentences")
    vectorizer = CountVectorizer(analyzer='word', stop_words=stop_words, lowercase=True)
    logging.info("Vectorizing...")
    vectorizer.fit(sentences_all)
    logging.info("Generating bags of words. Vocabulary="+str(len(vectorizer.vocabulary_))+" words")
    x_train = vectorizer.transform(sentences_train)
    x_test = vectorizer.transform(sentences_test)
    num_features = len(vectorizer.vocabulary_)
    return x_train, x_test, num_features


def train(x_train, y_train):
    # Fit the model
    # model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100)
    model = SVR()
    model.fit(x_train, y_train)
    return model


def evaluate(model, x_test, y_test):
    logging.info("Making predictions ...")
    # use the model to make predictions with the test data
    y_pred = model.predict(x_test)
    logging.info("Calculating accuracy")
    count_misclassified = (y_test != y_pred).sum()
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return count_misclassified, accuracy

