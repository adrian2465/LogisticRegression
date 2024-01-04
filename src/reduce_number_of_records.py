#!/usr/bin/python
import pandas as pd
from random import randrange
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
args_parser.add_argument("-pct", "--p", dest="percent", help="Percentage of lines to reduce to", required=True)
args = args_parser.parse_args()
filename = args.csv_path

filename_out = re.sub('\.csv', '_{}.csv'.format(args.percent), filename, flags=re.IGNORECASE)
if os.path.isfile(filename_out):
    logging.error("File '"+filename_out+"' already exists")
    exit(1)

# Load the data
logging.info("Loading " + filename)
pct = int(args.percent)
df = pd.read_csv(filename, escapechar='\\', delimiter=",", dtype={'LLTCODE': str}, skiprows=(lambda i: i>0 and randrange(100)>pct))  # Load dataframe from CSV using Pandas
logging.info("Total number of sentences loaded from disk: "+str(len(df)))
df.to_csv(filename_out, sep=',', quoting=csv.QUOTE_NONNUMERIC, index=False)

logging.info("Done")
exit(0)
