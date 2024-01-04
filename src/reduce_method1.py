import logging


def reduce(df, num_categories=0):
    # Count most populaced labels.
    df['count'] = df.groupby('LLTCODE')['LLTCODE'].transform('count')  # Add count column.
    categories = df[['LLTCODE', 'count']]
    categories = categories.drop_duplicates().sort_values(by='count', ascending=False)  # Count unique labels and sort by descending order of frequency
    logging.info("Total number of unique labels: " + str(len(categories)))
    if num_categories != 0:
        # Reduce to the requested number of categories.
        logging.info("Reducing number of labels to "+str(num_categories))
        categories = categories[:num_categories]

    logging.info("Filtering sentences with LLTCODES that are not in the list of categories.")
    df = df[df.LLTCODE.isin(categories.LLTCODE)]
    logging.info("Number of sentences remaining whose LLTCODEs match the categories: "+str(len(df)))

    logging.info("Transforming data to keep only alphanumeric")
    # Remove non-alphanumeric characters from inputs
    df = df.replace('[^a-zA-Z0-9 ]', '', regex=True)
    categories.reset_index(drop=True, inplace=True)

    return categories, df

