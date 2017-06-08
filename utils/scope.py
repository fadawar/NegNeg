import random
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pos_to_features import default_values, process_pos
from utils.common import add_prefix, detect_prefixes_and_particles, NONE_REPR, print_score, show_roc_curve
from utils.dataset import get_datasets, dataset_name, load_dataset


def add_nan(token_list, lemma_list, pos_list, is_in_scope_list, is_negator_list, dist_start_list, dist_end_list,
            dist_comma_left_list, dist_comma_right_list, dist_neg_signal_list, comma_between_list, negator_pos_list):
    token_list.append(NONE_REPR)
    token_list.append(NONE_REPR)
    lemma_list.append(NONE_REPR)
    lemma_list.append(NONE_REPR)
    pos_list.append(NONE_REPR)
    pos_list.append(NONE_REPR)
    is_in_scope_list.append(False)
    is_in_scope_list.append(False)
    is_negator_list.append(False)
    is_negator_list.append(False)
    dist_start_list.append(-1)
    dist_start_list.append(-1)
    dist_end_list.append(-1)
    dist_end_list.append(-1)
    dist_comma_left_list.append(-1)
    dist_comma_left_list.append(-1)
    dist_comma_right_list.append(-1)
    dist_comma_right_list.append(-1)
    dist_neg_signal_list.append(-1)
    dist_neg_signal_list.append(-1)
    comma_between_list.append(False)
    comma_between_list.append(False)
    negator_pos_list.append(NONE_REPR)
    negator_pos_list.append(NONE_REPR)


def create_semi_dataset(path):
    token_list = []
    lemma_list = []
    pos_list = []
    is_in_scope_list = []
    is_negator_list = []
    dist_start_list = []
    dist_end_list = []
    dist_comma_left_list = []
    dist_comma_right_list = []
    dist_neg_signal_list = []
    comma_between_list = []  # is comma between current word and negation signal?
    negator_pos_list = []

    root = ET.parse(path).getroot()
    sentences = root.getchildren()
    random.shuffle(sentences)

    for sentence in sentences:
        distance_from_end = len(sentence)
        negators = sentence.findall('./word[@negator]')

        for negator in negators:
            negator_position = sentence.getchildren().index(negator)
            add_nan(token_list, lemma_list, pos_list, is_in_scope_list, is_negator_list, dist_start_list, dist_end_list,
                    dist_comma_left_list, dist_comma_right_list, dist_neg_signal_list, comma_between_list,
                    negator_pos_list)

            for step, token in enumerate(sentence):
                token_list.append(token.text)
                lemma_list.append(token.attrib['lemma'])
                pos_list.append(token.attrib['pos'])
                is_in_scope_list.append(is_in_scope(token, negator))
                is_negator_list.append(token is negator)
                dist_start_list.append(step)
                dist_end_list.append(distance_from_end - step - 1)
                left_c, right_c = distance_commas(token, sentence)
                dist_comma_left_list.append(step - left_c)
                dist_comma_right_list.append(right_c - step)
                dist_neg_signal_list.append(negator_position - step)
                comma_between_list.append(step < right_c < negator_position or negator_position < left_c < step)
                negator_pos_list.append(negator.attrib['pos'])

            add_nan(token_list, lemma_list, pos_list, is_in_scope_list, is_negator_list, dist_start_list, dist_end_list,
                    dist_comma_left_list, dist_comma_right_list, dist_neg_signal_list, comma_between_list,
                    negator_pos_list)

    df_start = pd.DataFrame.from_dict({
        'token': token_list,
        'lemma': lemma_list,
        'POS': pos_list,
        'is_in_scope': is_in_scope_list,
        'is_negator': is_negator_list,
        'dist_start': dist_start_list,
        'dist_end': dist_end_list,
        'dist_comma_left': dist_comma_left_list,
        'dist_comma_right': dist_comma_right_list,
        'dist_neg_signal': dist_neg_signal_list,
        'comma_between': comma_between_list,
        'negator_pos': negator_pos_list,
    })

    # Split dataset into train and test sets
    split = 0.75
    train = df_start[:int(split * len(df_start))]
    test = df_start[int(split * len(df_start)):]
    X_train = train[['token', 'lemma', 'POS', 'dist_start', 'is_in_scope', 'is_negator', 'dist_end',
                     'dist_comma_left', 'dist_comma_right', 'dist_neg_signal', 'comma_between', 'negator_pos']]
    X_test = test[['token', 'lemma', 'POS', 'dist_start', 'is_in_scope', 'is_negator', 'dist_end',
                   'dist_comma_left', 'dist_comma_right', 'dist_neg_signal', 'comma_between', 'negator_pos']]
    return X_train, X_test


def find_element(sentence, text_to_find):
    for i, el in enumerate(sentence):
        if el.text == text_to_find:
            yield i


def distance_commas(token, sentence):
    commas = list(find_element(sentence, ','))
    position = sentence.getchildren().index(token)
    if len(commas) == 0:
        return -1, -1
    if position <= commas[0]:
        return 0, commas[0]
    elif position >= commas[-1]:
        return commas[-1], len(sentence)-1
    for first, second in zip(commas, commas[1:]):
        if first <= position <= second:
            return first, second


def is_in_scope(token, negator):
    if 'scope' in token.attrib.keys():
        if negator.attrib['id'] in token.attrib['scope'].split(','):
            return True
    return False


def create_columns_names(X_train, vect_lemma, add_bow):
    # create list with the names of columns in dataframe
    pos_feat_names = list(default_values.keys())
    column_names = ['token', 'lemma', 'POS'] + \
                   ['has_sk_prefix', 'has_int_prefix', 'is_particle',
                    'word_without_prefix_exist', 'start_with_ne'] + \
                   ['is_in_scope', 'is_negator', 'dist_start', 'dist_end', 'dist_comma_left'] + \
                   ['dist_comma_right', 'dist_neg_signal', 'comma_between']
    column_names += list(add_prefix('word1', pos_feat_names)) + \
                    list(add_prefix('word2', pos_feat_names)) + \
                    list(add_prefix('word3', pos_feat_names)) + \
                    list(add_prefix('word4', pos_feat_names)) + \
                    list(add_prefix('word5', pos_feat_names))
    column_names += list(add_prefix('negator', pos_feat_names))
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
        prefices_and_particles = np.array(detect_prefixes_and_particles(dataframe.loc[index + 2]['lemma']))

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

        # POS
        dtm_pos1 = list(process_pos(row['POS']).values())
        dtm_pos2 = list(process_pos(dataframe.loc[index + 1]['POS']).values())
        dtm_pos3 = list(process_pos(dataframe.loc[index + 2]['POS']).values())
        dtm_pos4 = list(process_pos(dataframe.loc[index + 3]['POS']).values())
        dtm_pos5 = list(process_pos(dataframe.loc[index + 4]['POS']).values())
        negator_pos = list(process_pos(row['negator_pos']).values())
        all_things = np.concatenate([
            [dataframe.loc[index + 2]['token'],
             dataframe.loc[index + 2]['lemma'],
             dataframe.loc[index + 2]['POS']],
            prefices_and_particles,
            [dataframe.loc[index + 2]['is_in_scope'],
             dataframe.loc[index + 2]['is_negator'],
             dataframe.loc[index + 2]['dist_start'],
             dataframe.loc[index + 2]['dist_end'],
             dataframe.loc[index + 2]['dist_comma_left'],
             dataframe.loc[index + 2]['dist_comma_right'],
             dataframe.loc[index + 2]['dist_neg_signal'],
             dataframe.loc[index + 2]['comma_between']],
            dtm_pos1,
            dtm_pos2,
            dtm_pos3,
            dtm_pos4,
            dtm_pos5,
            negator_pos,
            dtm_lemma,
        ])
        feautures_list.append(all_things)
    return feautures_list


def show_metrics_on_all_datasets(model, suffix, algorithm, with_all=False):
    for dataset in get_datasets(with_all):
        print('============================================')
        print(dataset_name(dataset))
        print('============================================')

        X_train, X_test, Y_train, Y_test = load_dataset(dataset, suffix, ['token', 'lemma', 'POS', 'is_in_scope'])
        X = pd.concat([X_train, X_test], axis=0)
        Y = pd.concat([Y_train, Y_test], axis=0)
        y_predicted = model.predict(X)
        y_true = Y.is_in_scope

        # Print basic metrics
        print_score(y_predicted, y_true)

        # Show ROC curve
        show_roc_curve(y_predicted, y_true, save_name='{}-{}-{}.svg'.format(suffix, dataset_name(dataset), algorithm))

