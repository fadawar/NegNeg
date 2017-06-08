import random
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pos_to_features import default_values, process_pos
from utils.common import add_prefix, detect_prefixes_and_particles, NONE_REPR, print_score, show_roc_curve
from utils.dataset import get_datasets, dataset_name, load_dataset


def add_nan(token_list, lemma_list, pos_list, is_negation_list):
    token_list.append(NONE_REPR)
    token_list.append(NONE_REPR)
    lemma_list.append(NONE_REPR)
    lemma_list.append(NONE_REPR)
    pos_list.append(NONE_REPR)
    pos_list.append(NONE_REPR)
    is_negation_list.append(0)
    is_negation_list.append(0)


def create_semi_dataset(path):
    token_list = []
    lemma_list = []
    pos_list = []
    is_negation_list = []

    root = ET.parse(path).getroot()
    sentences = root.getchildren()
    random.shuffle(sentences)

    for sentence in sentences:
        add_nan(token_list, lemma_list, pos_list, is_negation_list)
        for token in sentence:
            token_list.append(token.text)
            lemma_list.append(token.attrib['lemma'])
            pos_list.append(token.attrib['pos'])
            is_negation_list.append(int('negator' in token.attrib))
        add_nan(token_list, lemma_list, pos_list, is_negation_list)

    df_start = pd.DataFrame.from_dict({
        'token': token_list,
        'lemma': lemma_list,
        'POS': pos_list,
        'is_negation': is_negation_list,
    })

    # Split dataset into train and test sets
    split = 0.75
    train = df_start[:int(split * len(df_start))]
    test = df_start[int(split * len(df_start)):]
    X_train = train[['token', 'lemma', 'POS', 'is_negation']]
    X_test = test[['token', 'lemma', 'POS', 'is_negation']]
    return X_train, X_test


def create_columns_names(X_train, vect_lemma, add_bow):
    # create list with the names of columns in dataframe
    pos_feat_names = list(default_values.keys())
    column_names = ['token', 'lemma', 'POS', 'is_negation'] + \
                   ['has_sk_prefix', 'has_int_prefix', 'is_particle',
                    'word_without_prefix_exist', 'start_with_ne']
    column_names += list(add_prefix('word1', pos_feat_names)) + \
                    list(add_prefix('word2', pos_feat_names)) + \
                    list(add_prefix('word3', pos_feat_names)) + \
                    list(add_prefix('word4', pos_feat_names)) + \
                    list(add_prefix('word5', pos_feat_names))
    # Add bag of words
    if add_bow:
        lemma_feat_names = list(vect_lemma.get_feature_names())
        column_names += list(add_prefix('word1', lemma_feat_names)) + \
                        list(add_prefix('word2', lemma_feat_names)) + \
                        list(add_prefix('word3', lemma_feat_names)) + \
                        list(add_prefix('word4', lemma_feat_names)) + \
                        list(add_prefix('word5', lemma_feat_names))
    return column_names


def create_features_list(dataframe, vect_lemma, add_bow):
    feautures_list = []
    for index, row in dataframe.iterrows():
        if index >= dataframe.iloc[-4].name:
            break
        # Prefixes + Particles
        prefices_and_particles = np.array(detect_prefixes_and_particles(dataframe.loc[index+2]['lemma']))
        # POS
        dtm_pos1 = list(process_pos(row['POS']).values())
        dtm_pos2 = list(process_pos(dataframe.loc[index+1]['POS']).values())
        dtm_pos3 = list(process_pos(dataframe.loc[index+2]['POS']).values())
        dtm_pos4 = list(process_pos(dataframe.loc[index+3]['POS']).values())
        dtm_pos5 = list(process_pos(dataframe.loc[index+4]['POS']).values())

        dtm_lemma = []
        if add_bow:
            # LEMMA
            dtm_lemma1 = vect_lemma.transform([row['lemma']])
            dtm_lemma2 = vect_lemma.transform([dataframe.loc[index+1]['lemma']])
            dtm_lemma3 = vect_lemma.transform([dataframe.loc[index+2]['lemma']])
            dtm_lemma4 = vect_lemma.transform([dataframe.loc[index+3]['lemma']])
            dtm_lemma5 = vect_lemma.transform([dataframe.loc[index+4]['lemma']])
            dtm_lemma = np.concatenate([
                dtm_lemma1.toarray()[0],
                dtm_lemma2.toarray()[0],
                dtm_lemma3.toarray()[0],
                dtm_lemma4.toarray()[0],
                dtm_lemma5.toarray()[0],
            ])

        all_things = np.concatenate([
            [dataframe.loc[index+2]['token']],
            [dataframe.loc[index+2]['lemma']],
            [dataframe.loc[index+2]['POS']],
            [dataframe.loc[index+2]['is_negation']],
            prefices_and_particles,
            dtm_pos1,
            dtm_pos2,
            dtm_pos3,
            dtm_pos4,
            dtm_pos5,
            dtm_lemma,
        ])
        feautures_list.append(all_things)
    return feautures_list


def show_metrics_on_all_datasets(model, suffix, algorithm, with_all=False):
    for dataset in get_datasets(with_all):
        print('============================================')
        print(dataset_name(dataset))
        print('============================================')

        X_train, X_test, Y_train, Y_test = load_dataset(dataset, suffix, ['token', 'lemma', 'POS', 'is_negation'])
        X = pd.concat([X_train, X_test], axis=0)
        Y = pd.concat([Y_train, Y_test], axis=0)
        y_predicted = model.predict(X)
        y_true = Y.is_negation

        # Print basic metrics
        print_score(y_predicted, y_true)

        # Show ROC curve
        show_roc_curve(y_predicted, y_true, save_name='{}-{}-{}.svg'.format(suffix, dataset_name(dataset), algorithm))
