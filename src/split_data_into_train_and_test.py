#!/usr/bin/python
import pandas as pd
import logging
import numpy as np
from argparse import ArgumentParser
import os
import re
import csv


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Parse arguments
args_parser = ArgumentParser()
args_parser.add_argument("-csv", "--i", dest="csv_path", help="Consolidated CSV file created by getMedDRAData.sh script in GetMedCodingTrainingData project.", required=True)
args_parser.add_argument("-percent_test", "--p", dest="percent_test", help="Percentage (1-100) of data to split into test", required=True)
args = args_parser.parse_args()
percent_test = int(args.percent_test)
filename = args.csv_path

filename_test = re.sub('\.csv', '_test.csv', filename, flags=re.IGNORECASE)
filename_train = re.sub('\.csv', '_training.csv', filename, flags=re.IGNORECASE)
if os.path.isfile(filename_test):
    logging.error("File '"+filename_test+"' already exists")
    exit(1)
if os.path.isfile(filename_train):
    logging.error("File '"+filename_train+"' already exists")
    exit(1)

# Load the data
logging.info("Loading " + filename)
df = pd.read_csv(filename, escapechar='\\', delimiter=",", dtype={'LLTCODE': str})  # Load dataframe from CSV using Pandas
logging.info("Total number of sentences loaded from disk: "+str(len(df)))
num_test = int(percent_test*len(df)/100)
num_training = len(df)-num_test
sentences_test, sentences_train = np.split(df, [num_test])
sentences_test.to_csv(filename_test, sep=',', quoting=csv.QUOTE_NONNUMERIC, index=False)
sentences_train.to_csv(filename_train, sep=',', quoting=csv.QUOTE_NONNUMERIC, index=False)

logging.info("Test="+filename_test+" ("+str(num_test)+" rows)")
logging.info("Training="+filename_train+" ("+str(num_training)+" rows)")

logging.info("Done")
exit(0)
