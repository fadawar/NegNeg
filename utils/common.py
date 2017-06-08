import os
import pickle
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from pos_to_features import process_pos
from utils.dataset import get_datasets, dataset_name, load_dataset, DATASET_DIR

IMAGES_DIR = 'images'
MODELS_DIR = 'saved_models'
NONE_REPR = '__None__'  # representation of none word - at the beginning and end of sentence


def merge_dataset():
    subfolders = [f.path for f in os.scandir(DATASET_DIR) if f.is_dir()]
    for s in subfolders:
        sentences = []
        xml_files = [f.path for f in os.scandir(s) if f.is_file() and f.path.endswith('.xml') and not f.path.endswith('merged.xml')]
        for x in xml_files:
            with open(x) as f:
                root = ET.parse(f).getroot()
                sentences += root.getchildren()

        # save to file
        with open(os.path.join(s, 'merged.xml'), 'wb') as f:
            root = ET.Element('document')
            tree = ET.ElementTree(root)
            for sentence in sentences:
                root.append(sentence)
            tree.write(f, encoding='utf-8', xml_declaration=True)


def test_pos_processing():
    datasets = get_datasets()
    for dataset in datasets:
        with open(dataset) as f:
            root = ET.parse(f).getroot()
            sentences = root.getchildren()
            for sentence in sentences:
                try:
                    for element in sentence:
                        pos = element.attrib['pos']
                        process_pos(pos)
                except Exception as e:
                    print('SENTENCE: {}\nEXCEPTION: {}'.format(sentence.attrib['text'], e))


def print_score(y_pred, y_true):
    print('Accuracy: {}'.format(metrics.accuracy_score(y_true, y_pred)))
    print('Precision: {}'.format(metrics.precision_score(y_true, y_pred)))
    print('Recall: {}'.format(metrics.recall_score(y_true, y_pred)))
    print('F1 score: {}'.format(metrics.f1_score(y_true, y_pred)))
    print('Classification report:')
    print(metrics.classification_report(y_true, y_pred))


def show_roc_curve(y_pred, y_true, save_name=None):
    import matplotlib.pyplot as plt
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    if save_name:
        plt.savefig('{}/{}'.format(IMAGES_DIR, save_name), transparent=True)
    plt.show()


def save_model(model, name):
    with open('{}/{}'.format(MODELS_DIR, name), 'wb') as fout:
        pickle.dump(model, fout)


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


if __name__ == '__main__':
    pass
    # print(merge_dataset())
