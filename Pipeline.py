from DataIngester import DataIngester
from Preprocesser import Preprocesser
from Segregator import Segregator

class Pipeline(object):
    def __init__(self, model, train_file_path, predict_file_path):
        self.model = model
        self.filepath = {
            'train'   : train_file_path,
            'predict' : predict_file_path
        }

    def run_train_pipeline(self):
        # Data Ingestion
        print("\tIngesting Data")
        d = DataIngester(self.filepath['train'])
        d.drop_("key")
        data = (d.drop("category").to_numpy(), d.df['category'].to_numpy())

        # Preprocessing
        print("\tPreprocessing Data")
        p = Preprocesser(data)
        p.run()
        data = p.get_data()

        # Segregator
        print("\tSegregating Data")
        s = Segregator(1, 0.2, data[0], data[1])
        (x_train, x_test, y_train, y_test) = s.run()

        # Model Train/Evaluation
        print("\tTraining Model")
        m = self.model()
        m.train(x_train, y_train)
        acc = m.evaluate(x_test, y_test, verbosity=0)
        print("\tAccuracy: %.2f%%" % (acc * 100.0))
