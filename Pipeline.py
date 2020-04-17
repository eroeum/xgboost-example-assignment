from DataIngester import DataIngester
from Preprocesser import Preprocesser
from Segregator import Segregator
from Models.Model import Model

class Pipeline(object):
    def __init__(self, model, train_file_path, predict_file_path):
        self.model = model
        self.filepath = {
            'train'   : train_file_path,
            'predict' : predict_file_path
        }

    def run_train_pipeline(self):
        # Data Ingestion
        print("Ingesting Data")
        d = DataIngester(self.filepath['train'])
        d.drop_("key")
        data = (d.drop("category").to_numpy(), d.df['category'].to_numpy())

        # Preprocessing
        print("Preprocessing Data")
        p = Preprocesser(data)
        p.run()
        data = p.get_data()

        # Segregator
        print("Segregating Data")
        s = Segregator(1, 0.2, data[0], data[1])
        (x_train, x_test, y_train, y_test) = s.run()

        # Model Train/Evaluation
        print("Training Model")
        m = self.model()
        m.train(x_train, y_train)
        m.evaluate(x_test, y_test)

if __name__ == '__main__':
    p = Pipeline(Model, "data/train_data.csv", "data/test_data.csv")
    p.run_train_pipeline()
