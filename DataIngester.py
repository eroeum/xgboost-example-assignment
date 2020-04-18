"""
Data Ingester

Ingests data from csv files.
Operations added for dropping columns
"""

import pandas as pd

class DataIngester(object):
    def __init__(self, filepath):
        # Read CSV and set appropriate data values
        self.filepath  = filepath
        self.df = pd.read_csv(filepath,
            dtype = {'category': object})
        # Create appropriate labels
        if 'category' in self.df.columns:
            self.df['category'] = self.df['category'].apply(lambda x: int(x[-1]))

    # Implicit Row Drop
    def drop_(self, column):
        self.df = self.df.drop([column], axis=1)

    # Explicit Row Drop
    def drop(self, column):
        return self.df.drop([column], axis=1)

# Test Code
if __name__ == '__main__':
    d = DataIngester("data/train_data.csv")
    d.drop_("key")
    x = d.drop("category").to_numpy()
    y = d.df['category'].to_numpy()
