import os
from src.model_classification import Model_SVC

Model = Model_SVC()
dataset = Model.load_data_train(os.path.abspath("./data/classification_data.xlsx"))

Model.train(dataset)
Model.save()
