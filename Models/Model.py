from sklearn.metrics import accuracy_score
import xgboost as xgb

class Model(object):
    def __init__(self):
        self.model = self._init_model()

    def _init_model(self):
        return xgb.XGBClassifier()

    def train(self, x, y):
        self.model.fit(x, y)

    def evaluate(self, x, y, verbosity=1):
        pred = self.model.predict(x)
        acc = accuracy_score(y, pred)
        if verbosity > 0:
            print("Accuracy: %.2f%%" % (acc * 100.0))
        return acc
