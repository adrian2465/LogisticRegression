# Read input CSV, sort labels from most frequent to least frequent, then take top N labels (categories)
# Rewrite CSV such that only lines containing one of the N labels are produced.
from argparse import ArgumentParser
import pandas as pd
import os
import re
import csv
import logging
from reduce_method1 import reduce

Test_size = 0.2  # Size of test sample in percentage of total data. Training sample will be reduced to 1-Test_size.

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

filename_new = re.sub('\.csv', '.'+str(num_labels)+'.csv', filename, flags=re.IGNORECASE)
df.to_csv(filename_new, sep=',', quoting=csv.QUOTE_NONNUMERIC, index=False)
