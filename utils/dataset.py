import os
import sys
import unicodecsv as csv
import logging
import pandas as pd


# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create STDERR handler
handler = logging.StreamHandler(sys.stderr)
# Create formatter and add it to the handler
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# Set STDERR handler as the only handler
logger.handlers = [handler]


DATASET_DIR = 'dataset'
MERGED_FILE_NAME = 'merged.xml'


def get_datasets(with_all=False):
    def check_name(file):
        if file.is_dir():
            if not with_all:
                return not file.path.endswith('all')
            return True
        return False

    subfolders = [f.path for f in os.scandir(DATASET_DIR) if check_name(f)]
    return [os.path.join(s, MERGED_FILE_NAME) for s in subfolders]


def dataset_name(path):
    return path.split('/')[-2]


def remove_extension(path):
    return path.split('.')[0]


def preprocess_dataset(dataset_file, suffix, add_bow, functions, vect_lemma=None):
    """
    add_bow - create dataset with bag of words features
    functions = {
        'create_semi_dataset',
        'create_vectorizer',
        'create_columns_names',
        'create_features_list',
    }
    """
    logger.info('***** Create dataset {} *****'.format(dataset_name(dataset_file)))
    ouput_name = '{}.csv'.format(remove_extension(dataset_file))
    logger.info('[START] Creating semi dataset')
    X_train, X_test = functions['create_semi_dataset'](dataset_file)
    if add_bow and not vect_lemma:
        vect_lemma = functions['create_vectorizer'](X_train)
    column_names = functions['create_columns_names'](X_train, vect_lemma, add_bow)
    logger.info('[START] Creating features list - train')
    X_train_features_list = functions['create_features_list'](X_train, vect_lemma, add_bow)
    logger.info('[START] Creating features list - test')
    X_test_features_list = functions['create_features_list'](X_test, vect_lemma, add_bow)
    logger.info('[START] Saving CSV')
    save_to_csv('{}-{}-train'.format(ouput_name, suffix), column_names, X_train_features_list)
    save_to_csv('{}-{}-test'.format(ouput_name, suffix), column_names, X_test_features_list)
    logger.info('[FINISH] Saving CSV')
    return vect_lemma


def save_to_csv(filename, header_row, feautures_list):
    with open(filename, 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
        wr.writerow(header_row)
        for row in feautures_list:
            wr.writerow(row)


def load_dataset(dataset_file, suffix, columns_for_Y):
    """
    Loads dataset. Return dataframes with only training features (X_something) and
    dataframes with correct answer and leftover features (these are only for manual testing)
    """
    ouput_name = '{}.csv'.format(remove_extension(dataset_file))
    train_all = pd.read_csv('{}-{}-train'.format(ouput_name, suffix))
    test_all = pd.read_csv('{}-{}-test'.format(ouput_name, suffix))
    X_train = train_all[train_all.columns.difference(columns_for_Y)]
    X_test = test_all[test_all.columns.difference(columns_for_Y)]

    # X_train, X_test, Y_train, Y_test
    return X_train, X_test, train_all[columns_for_Y], test_all[columns_for_Y]
