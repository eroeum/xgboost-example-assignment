from Pipeline import Pipeline
import Models

def main():
    train_data = "data/train_data.csv"
    test_data  = "data/test_data.csv"

    models = {
        "Standard" : Models.Model
    }

    for name, model in models.items():
        print(name)
        p = Pipeline(model, train_data, test_data)
        p.run_train_pipeline()


if __name__ == '__main__':
    main()
