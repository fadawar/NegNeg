import json
import pickle
import requests
import pandas as pd
from utils import signal, scope
from utils.common import NONE_REPR
from xml.etree.ElementTree import Element, SubElement

POS_URL = 'http://nlp.bednarik.top/tagger/json'
POS_LEMMA_URL = 'http://nlp.bednarik.top/lemmatizer/json'

with open('saved_models/model-RandomForest-sig.pkl', 'rb') as f:
    model_sig = pickle.load(f)

with open('saved_models/model-RandomForest-sco.pkl', 'rb') as f:
    model_sco = pickle.load(f)


def find_negation(text):
    data = get_pos_and_lemma(text)
    for sentence in data:
        find_negation_signal(sentence)
        if has_negation_signal(sentence):
            find_negation_scope(sentence)
    return data


def find_negation_signal(data):
    df = create_semi_dataset_sig(data)
    fl = signal.create_features_list(df, None, False)
    df2 = pd.DataFrame(fl, columns=signal.create_columns_names(None, None, False))
    X = df2[df2.columns.difference(['token', 'lemma', 'POS', 'is_negation'])]
    predicted = model_sig.predict(X)
    for i, d in enumerate(data):
        d['negator'] = int(predicted[i])


def has_negation_signal(data):
    return any((d['negator'] for d in data))


def find_negation_scope(data):
    xml = create_xml_sco(data)
    df = scope.csd_body(xml)
    fl = scope.create_features_list(df, None, False)
    df2 = pd.DataFrame(fl, columns=scope.create_columns_names(None, None, False))
    X = df2[df2.columns.difference(['token', 'lemma', 'POS', 'is_in_scope'])]
    predicted = model_sco.predict(X)
    for i, d in enumerate(data):
        d['scope'] = int(predicted[i])


def get_pos_and_lemma(sentence):
    r = requests.post(POS_LEMMA_URL, data={'input': sentence, 'method': 'WITHPOS'})
    j = json.loads(r.content.decode('utf-8'))
    sentences = []
    for sentence in j['sentences']:
        sentences.append(
            [{'pos': d['tag'], 'word': d['text'], 'lemma': d['lemma']} for d in sentence['tokens']]
        )
    return sentences


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
