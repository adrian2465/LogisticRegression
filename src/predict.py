from argparse import ArgumentParser
import pandas as pd
import logging
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split

Test_size = 0.2  # Size of test sample in percentage of total data. Training sample will be reduced to 1-Test_size.
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Parse arguments
args_parser = ArgumentParser()
args_parser.add_argument("--incsv", "-i", dest="incsv", help="CSV file containing test data", required=True)
args_parser.add_argument("--model", "-m", dest="model", help="suffix for .vectorizer and .model files", required=True)
args = args_parser.parse_args()

# Load model
model_filename=args.model+".model"
print("Loading model"+model_filename)
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load model
vectorizer_filename=args.model+".vectorizer"
print("Loading vectorizer"+model_filename)
with open(vectorizer_filename, 'rb') as file:
    vectorizer = pickle.load(file)

# Read input CSV (Test file)
print("Loading CSV"+args.incsv)
df = pd.read_csv(args.incsv, escapechar='\\', delimiter=",", dtype={'LLTCODE': str})  # Load dataframe from CSV using Pandas

# sentences_train, sentences_test, y_train_label, y_test_label = train_test_split(df.LITERAL.astype('U'), df.LLTCODE.astype('U'), test_size=Test_size)
sentences_test = df.LITERAL.astype('U')
y_test_label = df.LLTCODE.astype('U')

x_test = vectorizer.transform(sentences_test)
y_pred = model.predict(x_test)
score = metrics.accuracy_score(y_test_label,y_pred)

# newdf = pd.DataFrame({'y_pred': y_pred[:,0], 'y_test_label': y_test_label[:,0], 'sentence':sentences_test[:,0]})

hit=0
print ("ACCURACY RESULTS USING CSV {} and MODEL {}".format(args.incsv,args.model))
print ("hit or miss,y_pred[i],y_test_label[i],sentences_test[i]")
y_test_label_array=y_test_label.array
sentences_test_array=sentences_test.array
for i in range(len(y_pred)):
  if (y_pred[i] != y_test_label_array[i]):
    print ("miss",",",y_pred[i],",",y_test_label_array[i],",",sentences_test_array[i])
  else:
    hit=hit+1
    print ("hit",",",y_pred[i],",",y_test_label_array[i],",",sentences_test_array[i])

print("Total test sentences: {} Total test labels: {}, Score: {:.2%}".format(len(y_pred),len(vectorizer.vocabulary_),score))
