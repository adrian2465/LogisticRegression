from argparse import ArgumentParser
import pandas as pd
import os
import logging
import pickle
from logistic_regression_BOW import generate_model
# from SVM_BOW import generate_model
from reduce_method1 import reduce
from sklearn.model_selection import train_test_split


Test_size = 0.2  # Size of test sample in percentage of total data. Training sample will be reduced to 1-Test_size.
# Model_type = "svm"
Model_type = "lr"

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Parse arguments
args_parser = ArgumentParser()
args_parser.add_argument("-csv", "--i", dest="csv_path", help="Consolidated CSV file created by getMedDRAData.sh script in GetMedCodingTrainingData project.", required=True)
args_parser.add_argument("-numcat", "--c", dest="num_categories", help="Number of categories to which to map. 0=all", required=False, default=0)
args = args_parser.parse_args()
num_labels = int(args.num_categories)

logging.info("PID="+str(os.getpid()))
filename = args.csv_path
# Load the data
logging.info("Loading " + filename)
df = pd.read_csv(filename, escapechar='\\', delimiter=",", dtype={'LLTCODE': str})  # Load dataframe from CSV using Pandas
logging.info("Total number of sentences loaded from disk: "+str(len(df)))

# Reduce the feature space and data size.
categories, df = reduce(df, num_labels)

# Convert data labels to consecutive integers.
# Example how to get index of matching row. print(df[df['LLTCODE']=="10051775"])
sentences = df.LITERAL.astype('U')

# Split data into training and test.
sentences_train, sentences_test, y_train_label, y_test_label = train_test_split(sentences, df.LLTCODE.astype('U'), test_size=Test_size)

# Generate a model
vectorizer, model = generate_model(sentences, sentences_train, sentences_test, y_train_label, y_test_label, num_labels, filename)

# Save model
vec_filename = args.csv_path+'.'+Model_type+'.'+args.num_categories+'.vectorizer'
model_filename = args.csv_path+'.'+Model_type+'.'+args.num_categories+'.model'

logging.info('Saving vectorizer to '+vec_filename)
pickle.dump(vectorizer, open(vec_filename, 'wb'))

logging.info('Saving model to '+model_filename)
pickle.dump(model, open(model_filename, 'wb'))

logging.info("Done")
exit(0)
