"""
Helper module for handling the generated Datasets
"""
import os
import pickle
from sklearn.model_selection import train_test_split
from consts import EXPERT_DATA_DIR
dataset_dir = EXPERT_DATA_DIR


def load_dataset(dataset_name, dataset_dir):
    """
    Loads the pickle files saved in run_expert.py

    :param dataset_name: str
    :return: dict
    """
    print("loading dataset %s" % dataset_name)
    pickle_filename = '%s.pkl' % dataset_name
    with open(os.path.join(dataset_dir, pickle_filename), 'rb') as f:
        expert_data = pickle.load(f)
    return expert_data


def get_datasets(dataset_name, dataset_dir):
    """
    Loads and return a 4-tuple of np.arrays of train/validation features and labels (X_train, X_test, y_train, y_test)

    :param dataset_name:
    :return: tuple (4 np.arrays)
    """
    expert_data = load_dataset(dataset_name=dataset_name, dataset_dir=dataset_dir)
    X = expert_data['observations']
    y = expert_data['actions']

    # Use 90% of the samples as the train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    print("Domain name: %s" % dataset_name)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test
