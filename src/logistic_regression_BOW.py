from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import logging
import os.path
import csv

stop_words = ['the', 'if', 'and', 'but', 'or', 'with', 'for', 'not', 'with', 'by', 'indicating', 'indicates', 'noted']


def generate_model(sentences, sentences_train, sentences_test, y_train_label, y_test_label, num_labels, filename):
    algorithm = "Logistic Regression Bag of Words"
    num_sentences = len(sentences)
    logging.info("Generating model using logistic_regression_BOW")
    # Vectorize
    x_train, x_test, num_features, vectorizer = vectorize(sentences, sentences_train, sentences_test)

    # Train
    model = train(x_train, y_train_label)

    # Evaluate
    count_misclassified, accuracy = evaluate(model, sentences_test, x_test, y_test_label,vectorizer)
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

    return vectorizer, model


def vectorize(sentences_all, sentences_train, sentences_test):
    logging.info("Stop words: "+str(stop_words))
    # Prepare for training. Fit the sentences to a vocabulary/vectorizer....
    logging.info("Training set=" + str(len(sentences_train)) + ", test set="+str(len(sentences_test))+" sentences")
    vectorizer = CountVectorizer(analyzer='word', stop_words=stop_words)
    logging.info("Vectorizing...")
    vectorizer.fit(sentences_all)
    logging.info("Generating bags of words. Vocabulary="+str(len(vectorizer.vocabulary_))+" words")
    x_train = vectorizer.transform(sentences_train)
    x_test = vectorizer.transform(sentences_test)
    num_features = len(vectorizer.vocabulary_)
    return x_train, x_test, num_features, vectorizer


def train(x_train, y_train_label):
    # Fit the model
    # lbfgs — Stands for Limited-memory Broyden–Fletcher–Goldfarb–Shanno. 
    # It approximates the second derivative matrix updates with gradient evaluations. 
    # It stores only the last few updates, so it saves memory. It isn't super fast with large data sets. 
    # It will be the default solver as of Scikit-learn version 0.22.
    model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100)
    model.fit(x_train, y_train_label)
    return model


def evaluate(model, sentences_test, x_test, y_test_label,vectorizer):
    logging.info("Making predictions ...")
    # use the model to make predictions with the test data
    y_pred = model.predict(x_test)
    logging.info("Calculating accuracy")
    count_misclassified = (y_test_label != y_pred).sum()
    accuracy = metrics.accuracy_score(y_test_label, y_pred)
    print("Accuracy_score={}".format(accuracy))
    hit=0
    print ("hit or miss,y_pred[i],y_test_label[i],sentences_test[i]")
    y_test_label_array=y_test_label.array
    sentences_test_array=sentences_test.array
    for i in range(len(y_pred)):
      if (y_pred[i] != y_test_label_array[i]):
        print ("miss",",",y_pred[i],",",y_test_label_array[i],",",sentences_test_array[i])
      else:
        hit=hit+1
        print ("hit",",",y_pred[i],",",y_test_label_array[i],",",sentences_test_array[i])

    print("Total test sentences: {} Total test labels: {}, Score: {:.2%}".format(len(y_pred),len(vectorizer.vocabulary_),accuracy))
    print("Hits: {}. Total: {}".format(hit,len(y_pred)))
    return count_misclassified, accuracy

