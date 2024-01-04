# Count LLT codes.  Differs from the Almaden count_llt.py because this uses CSV as input, rather than CLADs
import logging
from argparse import ArgumentParser
import pandas as pd

def get_counts(df):
    # Count most populaced labels.
    df['count'] = df.groupby('LLTCODE')['LLTCODE'].transform('count')  # Add count column.
    categories = df[['LLTCODE', 'count']]
    categories = categories.drop_duplicates().sort_values(by='count', ascending=False)  # Count unique labels and sort by descending order of frequency
    categories.reset_index(drop=True, inplace=True)
    return categories

def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    # Parse arguments
    args_parser = ArgumentParser()
    args_parser.add_argument("-csv", "--i", dest="csv_path", help="Consolidated CSV", required=True)
    args = args_parser.parse_args()
    filename = args.csv_path
    # Load the data
    logging.info("Loading " + filename)
    df = pd.read_csv(filename, escapechar='\\', delimiter=",", dtype={'LLTCODE': str})  # Load dataframe from CSV using Pandas
    logging.info("Total number of sentences loaded from disk: "+str(len(df)))
    categories = get_counts(df)
    outfilename = filename+".counts.csv"
    categories.to_csv(outfilename, index=False)
    logging.info("Wrote results into "+outfilename)

if __name__ == "__main__":
    main()
