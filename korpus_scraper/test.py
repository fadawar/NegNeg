import pandas as pd
import xml.etree.ElementTree as ET

root = ET.parse('dataset.xml').getroot()

df = pd.DataFrame.from_records(
    [(token.text, token.attrib['lemma'], token.attrib['pos']) for token in root.iter('token')],
    columns=('token', 'lemma', 'pos')
)

print(df.head())
