import pandas as pd

class DataIngester(object):
    def __init__(self, filepath):
        self.filepath  = filepath
        self.df = pd.read_csv(filepath,
            dtype = {'category': object})
        if 'category' in self.df.columns:
            self.df['category'] = self.df['category'].apply(lambda x: int(x[-1]))

    def drop_(self, column):
        self.df = self.df.drop([column], axis=1)

    def drop(self, column):
        return self.df.drop([column], axis=1)

if __name__ == '__main__':
    d = DataIngester("data/train_data.csv")
    d.drop_("key")
    x = d.drop("category").to_numpy()
    y = d.df['category'].to_numpy()
