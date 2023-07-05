import numpy as np
import pandas as pd

from util import get_logger


class DataHandlerABC:

    def __init__(self, path, preprocess=True):
        self.logger = get_logger(__name__ + "." + self.__class__.__name__)

        self.data = pd.read_csv(path)
        self.logger.info(f"Loaded data from `{path}` successfully.")

        if preprocess:
            self.preprocess_data()

    def preprocess_data(self):
        pass

    def partition_data(self):
        return None


class Penguins(DataHandlerABC):

    def preprocess_data(self):
        # Replace Na values with the last valid value of their column
        self.data.fillna('pad', inplace=True)
        # Convert `male` to 1, `female` to 0
        self.data['gender'] = np.where(self.data['gender'] == 'male', 1, 0)

        # Convert `flipper_length_mm` to Cm
        self.data['flipper_length_mm'] = self.data['flipper_length_mm'] / 10
        # Convert `body_mass_g` to Kg
        self.data['body_mass_g'] = self.data['body_mass_g'] / 1000
        
        # Convert all values to fractions
        func = lambda x: x / 100
        self.data['flipper_length_mm'] = self.data['flipper_length_mm'].apply(func)
        self.data['bill_length_mm'] = self.data['bill_length_mm'].apply(func)
        self.data['bill_depth_mm'] = self.data['bill_depth_mm'].apply(func)
        self.data['body_mass_g'] = self.data['body_mass_g'].apply(func)

        # Convert species' names to 0-index classes
        y_label = "species"
        names = self.data[y_label].unique()
        labels = {names[i]: i for i in range(len(names))}
        self.data[y_label] = self.data[y_label].apply(lambda x: labels[x])

        self.logger.info("Data preprocessed successfully.")

    def partition_data(self):
        # Create empty DataFrames
        training = pd.DataFrame()
        testing = pd.DataFrame()

        y_label = "species"

        # Group by the label, then add 30 training rows to the training DataFrame
        # and 20 test rows to the testing frame
        for _, group in self.data.groupby(y_label):
            training = pd.concat([training, group.iloc[:30]], ignore_index=True)
            testing = pd.concat([testing, group.iloc[30:]], ignore_index=True)

        # Randomly shuffle the data
        training = training.sample(frac=1)
        testing = testing.sample(frac=1)

        # Return the feature columns and label column separately for each DataFrame, such that the return values
        # are training_feature_columns, training_label_column, testing_feature_columns, testing_label_coloumn
        # Inspired by scikit-learn train_test_split function
        return (
            training.loc[:, training.columns != y_label],
            training[y_label],
            testing.loc[:, testing.columns != y_label],
            testing[y_label]
        )


class MNIST(DataHandlerABC):

    def __init__(self, train_path, test_path, preprocess=True):
        self.logger = get_logger(__name__ + "." + self.__class__.__name__)

        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)

        self.logger.info(f"Loaded data from `{train_path}` and `{test_path}` successfully.")

        if preprocess:
            self.preprocess_data()

    def preprocess_data(self):
        # Convert pixel values from range [0, 255] to [0, 1]
        filt = self.train_data.columns != "label"
        func = lambda x: x / 255

        self.train_data.loc[:, filt] = self.train_data.loc[:, filt].applymap(func)
        self.test_data.loc[:, filt] = self.test_data.loc[:, filt].applymap(func)

        self.logger.info("Data preprocessed successfully.")

    def partition_data(self):
        y_label = "label"
        filt = self.train_data.columns != y_label

        x_train = self.train_data.loc[:, filt]
        y_train = self.train_data[y_label]

        x_test = self.test_data.loc[:, filt]
        y_test = self.test_data[y_label]

        return x_train, y_train, x_test, y_test


class Iris(DataHandlerABC):

    def preprocess_data(self):
        y_label = "class"

        filt = self.data.columns != y_label
        self.data.loc[:, filt] = self.data.loc[:, filt].applymap(lambda x: x / 10)
    
        names = self.data[y_label].unique()
        labels = {names[i]: i for i in range(len(names))}
        self.data[y_label] = self.data[y_label].apply(lambda x: labels[x])

        self.logger.info("Data preprocessed successfully.")

    def partition_data(self):
        # Create empty DataFrames
        training = pd.DataFrame()
        testing = pd.DataFrame()

        y_label = "class"

        # Group by the label, then add 30 training rows to the training DataFrame
        # and 20 test rows to the testing frame
        for _, group in self.data.groupby(y_label):
            training = pd.concat([training, group.iloc[:30]], ignore_index=True)
            testing = pd.concat([testing, group.iloc[30:]], ignore_index=True)

        # Randomly shuffle the data
        training = training.sample(frac=1)
        testing = testing.sample(frac=1)

        # Return the feature columns and label column separately for each DataFrame, such that the return values
        # are training_feature_columns, training_label_column, testing_feature_columns, testing_label_coloumn
        # Inspired by scikit-learn train_test_split function
        return (
            training.loc[:, training.columns != y_label],
            training[y_label],
            testing.loc[:, testing.columns != y_label],
            testing[y_label]
        )


DATASETS = [
    "Penguins",
    "MNIST",
    "Iris"
]
