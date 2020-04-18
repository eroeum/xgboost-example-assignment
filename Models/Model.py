"""
XGBoost model for other models to inherit
"""

from sklearn.metrics import accuracy_score
import xgboost as xgb

class Model(object):
    def __init__(self):
        """Calls init model to create actual model"""
        self.model = self._init_model()

    def _init_model(self):
        """Each child should override this to create its own"""
        return xgb.XGBClassifier()

    def train(self, x, y):
        """Train Model"""
        self.model.fit(x, y)

    def evaluate(self, x, y, verbosity=1):
        """Evaluate Accuracy"""
        pred = self.model.predict(x)
        acc = accuracy_score(y, pred)
        if verbosity > 0:
            print("Accuracy: %.2f%%" % (acc * 100.0))
        return acc

    def predict(self, x):
        """Predict Data"""
        pred = self.model.predict(x)
        return pred

    def write_predictions(self, df, predictions, filename):
        """Write Prediction to csv"""
        df.drop(df.columns.difference(['key']), 1, inplace=True)
        df['pred'] = predictions

        df["group_01"] = df['pred'] == 1
        df["group_02"] = df['pred'] == 2
        df["group_03"] = df['pred'] == 3
        df["group_04"] = df['pred'] == 4
        df["group_05"] = df['pred'] == 5
        df["group_06"] = df['pred'] == 6
        df["group_07"] = df['pred'] == 7
        df["group_08"] = df['pred'] == 8
        df["group_09"] = df['pred'] == 9

        df = df.astype('int32')
        df = df.drop(['pred'], axis=1)

        df.to_csv(filename, index=False)

        return df
