import json
import pickle
import requests
import pandas as pd
from utils import signal, scope
from utils.common import NONE_REPR
from xml.etree.ElementTree import Element, SubElement, tostring

POS_URL = 'http://nlp.bednarik.top/tagger/json'
POS_LEMMA_URL = 'http://nlp.bednarik.top/lemmatizer/json'


def find_negation(sentence):
    data = get_pos_and_lemma(sentence)
    # signal
    df = create_semi_dataset_sig(data)
    fl = signal.create_features_list(df, None, False)
    df2 = pd.DataFrame(fl, columns=signal.create_columns_names(None, None, False))
    X = df2[df2.columns.difference(['token', 'lemma', 'POS', 'is_negation'])]
    with open('saved_models/model-RandomForest-sig.pkl', 'rb') as f:
        model = pickle.load(f)
    predicted = model.predict(X)
    for i, d in enumerate(data):
        d['negator'] = predicted[i]
    # scope
    xml = create_xml_sco(data)
    df = scope.csd_body(xml)
    fl = scope.create_features_list(df, None, False)
    df2 = pd.DataFrame(fl, columns=scope.create_columns_names(None, None, False))
    X = df2[df2.columns.difference(['token', 'lemma', 'POS', 'is_in_scope'])]
    with open('saved_models/model-RandomForest-sco.pkl', 'rb') as f:
        model = pickle.load(f)
    predicted = model.predict(X)
    for i, d in enumerate(data):
        d['scope'] = predicted[i]
    return data


def get_pos(sentence):
    r = requests.post(POS_URL, data={'input': sentence})
    j = json.loads(r.content.decode('utf-8'))
    return [{'pos': d['tag'], 'word': d['text']} for d in j['sentences'][0]['tokens']]


def get_pos_and_lemma(sentence):
    r = requests.post(POS_LEMMA_URL, data={'input': sentence, 'method': 'WITHPOS'})
    j = json.loads(r.content.decode('utf-8'))
    return [{'pos': d['tag'], 'word': d['text'], 'lemma': d['lemma']} for d in j['sentences'][0]['tokens']]


def create_semi_dataset_sig(data):
    df = pd.DataFrame(columns=['token', 'lemma', 'POS', 'is_negation'])
    df.loc[0] = [NONE_REPR, NONE_REPR, NONE_REPR, 0]
    df.loc[1] = [NONE_REPR, NONE_REPR, NONE_REPR, 0]
    for i, row in enumerate(data, start=2):
        df.loc[i] = [row['word'], row['lemma'], row['pos'], 0]
    i = 2 + len(data)
    df.loc[i] = [NONE_REPR, NONE_REPR, NONE_REPR, 0]
    df.loc[i+1] = [NONE_REPR, NONE_REPR, NONE_REPR, 0]
    return df


# def create_semi_dataset_sco(data):
#     df = pd.DataFrame(columns=['token', 'lemma', 'POS', 'is_in_scope', 'is_negator', 'dist_start', 'dist_end',
#                                'dist_comma_left', 'dist_comma_right', 'dist_neg_signal', 'comma_between',
#                                'negator_pos'])
#     df.loc[0] = [NONE_REPR, NONE_REPR, NONE_REPR, 0, 0, -1, -1, -1, -1, -1, 0, NONE_REPR]
#     df.loc[1] = [NONE_REPR, NONE_REPR, NONE_REPR, 0, 0, -1, -1, -1, -1, -1, 0, NONE_REPR]
#     for i, row in enumerate(data, start=2):
#         df.loc[i] = [row['word'], row['lemma'], row['pos'], 0, ]
#     i = 2 + len(data)
#     df.loc[i] = [NONE_REPR, NONE_REPR, NONE_REPR, 0, 0, -1, -1, -1, -1, -1, 0, NONE_REPR]
#     df.loc[i+1] = [NONE_REPR, NONE_REPR, NONE_REPR, 0, 0, -1, -1, -1, -1, -1, 0, NONE_REPR]
#     return df


def create_xml_sco(data):
    doc = Element('document')
    sentence = SubElement(doc, 'sentence')
    for i, row in enumerate(data):
        word = SubElement(sentence, 'word')
        word.text = row['word']
        word.attrib['id'] = 'w{}'.format(i)
        word.attrib['lemma'] = row['lemma']
        word.attrib['pos'] = row['pos']
        if row['negator'] == 1:
            word.attrib['negator'] = str(1)
    return doc
