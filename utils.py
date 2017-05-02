import os
import xml.etree.ElementTree as ET
from pos_to_features import process_pos
from sklearn import metrics
import matplotlib.pyplot as plt


DATASET_DIR = 'dataset'


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


def get_datasets():
    subfolders = [f.path for f in os.scandir(DATASET_DIR) if f.is_dir()]
    return [os.path.join(s, 'merged.xml') for s in subfolders]


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


def print_score_for(model, df2_test, y_true):
    y_pred = model.predict(df2_test)

    print('Accuracy: {}'.format(metrics.accuracy_score(y_true, y_pred)))
    print('Precision: {}'.format(metrics.precision_score(y_true, y_pred)))
    print('Recall: {}'.format(metrics.recall_score(y_true, y_pred)))
    print('F1 score: {}'.format(metrics.f1_score(y_true, y_pred)))
    print(metrics.classification_report(y_true, y_pred))
    return y_true, y_pred


def show_roc_curve(y_true, y_pred):
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
    return plt


if __name__ == '__main__':
    print(test_pos_processing())
