"""
Driver Code

Runs driver code to run through training and prediction pipelines.
Example:
    python run.py
"""

from Pipeline import Pipeline
import Models
from notification import Notifier

def main():
    """Driver Function"""
    # Create Notificater (For SMS notification of status)
    # NOTE: Need to set environment variables before using
    # notifier = Notifier("+13013510464")

    # Path to data (Relative)
    train_data = "data/train_data.csv"
    test_data  = "data/test_data.csv"

    # List of model classes to run
    # Key: name
    # Value: Class of model
    models = {
        "New Standard" : Models.R_1
    }

    # Run through all model pipelines
    for name, model in models.items():
        # notifier.start()

        print(name)
        p = Pipeline(model, train_data, test_data)
        acc = p.run_train_pipeline()
        df = p.run_predict_pipeline()
        
        # notifier.end()
        # notifier.notify_acc(acc)

    return df

if __name__ == '__main__':
    df = main()
