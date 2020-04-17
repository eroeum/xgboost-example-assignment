import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

if __name__ == '__main__':
    # Import Data
    df = pd.read_csv("data/train_data.csv", dtype = {'category': object})
    df['category'] = df['category'].apply(lambda x: int(x[-1]))
    df = df.drop(["key"], axis=1)

    x = df.drop(['category'], axis=1).to_numpy()
    # y = np.reshape(df['category'].to_numpy(), (-1, 1))
    y = df['category'].to_numpy()

    # Segregator
    seed = 9
    split = 0.2
    x_train, x_test, y_train, y_test = train_test_split(x, y,
        test_size    = split,
        random_state = seed)

    # Model Train
    model = xgb.XGBClassifier(verbosity=1)
    model.fit(x_train, y_train)

    # Model Evaluation
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
