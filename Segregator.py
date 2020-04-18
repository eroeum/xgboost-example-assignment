"""
Segregates data into training and testing data

Basically just a big wrapper for train_test_split
"""

from sklearn.model_selection import train_test_split

class Segregator(object):
    def __init__(self, seed, split, x, y):
        self.seed = seed
        self.split = split
        self.x = x
        self.y = y

    def run(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y,
            test_size    = self.split,
            random_state = self.seed)

        return (x_train, x_test, y_train, y_test)
