import random
from datetime import datetime
import pandas as pd
import xml.etree.ElementTree as ET
# import csv
import unicodecsv as csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pos_to_features import default_values, process_pos


NONE_REPR = '__None__'  # representation of none word - at the beginning and end of sentence


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


# Prepare sets for features
prefix_sk = {'ne', 'bez', 'pa', 'roz', 'proti', 'polo', 'tiež', 'akoby', 'trochu', 'truc', 'mimo', }
prefix_int = {'pseudo', 'i', 'in', 'anti', 'kontra', 'a', 'an', 'ex', 'non', 'kvázi', 'hypo', 'de', 'dez', 'ex',
              'extra', }
particles = {'nie', 'figu', 'drevenú', 'figu', 'borovú', 'jalovú', 'figu', 'šušku', 'šušku', 'borovú', 'čerta', 'čerta',
             'starého', 'čerta', 'rohatého', 'čerta', 'pekelného', 'čerta', 'ušatého', 'čerta', 'strapatého', 'paroma',
             'paroma', 'starého', 'hroma', 'psiu', 'mater', 'horký', 'horkýtam', 'horkýže', 'horkýžetam', 'aleba',
             'ale', 'čo', 'kde', 'kdeže', 'kdeby', 'kdežeby', 'kdežetam', 'kdežebytam', 'čo', 'čože', 'čoby', 'čožeby',
             'ešte', 'čo', 'ešteže', 'čo', 'akurát', 'javeru', 'rozhodne', 'rovno', 'aký', 'akýže', 'akéže', }
slovak_dict = set(line.strip() for line in open('sk.dic'))


def start_with_prefix(word, prefixes):
    for p in prefixes:
        if word.startswith(p):
            return True, p
    return False, p


def detect_prefixes_and_particles(lemma):
    has_sk_prefix, p_sk = start_with_prefix(lemma, prefix_sk)
    has_int_prefix, p_int = start_with_prefix(lemma, prefix_int)
    is_particle = lemma in particles
    if has_sk_prefix:
        prefix = p_sk
    elif has_int_prefix:
        prefix = p_int
    else:
        prefix = None
    start_with_ne = False
    word_without_prefix_exist = False
    if prefix:
        word_without_prefix_exist = lemma[len(prefix):] in slovak_dict
        if word_without_prefix_exist and prefix == 'ne':
            start_with_ne = True
    return has_sk_prefix, has_int_prefix, is_particle, word_without_prefix_exist, start_with_ne


def add_prefix(prefix, iterable):
    for item in iterable:
        yield prefix + "_" + item


def create_vectorizer(X_train):
    # learn training data vocabulary, then use it to create a document-term matrix
    vect_lemma = CountVectorizer()
    vect_lemma.fit(X_train.lemma)
    return vect_lemma


def create_columns_names(X_train, vect_lemma, add_bow):
    # create list with the names of columns in dataframe
    lemma_feat_names = list(vect_lemma.get_feature_names())
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


def save_to_csv(filename, header_row, feautures_list):
    with open(filename, 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
        wr.writerow(header_row)
        for row in feautures_list:
            wr.writerow(row)
