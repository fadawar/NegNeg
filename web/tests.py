import pytest

from .negneg import find_negation, get_pos, get_pos_and_lemma, create_semi_dataset_sig

correct_xml = """<sentence text="Pavol dnes večer nepríde.">
    <word id="w1" scope="w4" lemma="Pavol" pos="PAis7">Pavol</word>
    <word id="w2" scope="w4" lemma="dnes" pos="SSms7">dnes</word>
    <word id="w3" scope="w4" lemma="večer" pos="Eu4">večer</word>
    <word id="w4" negator="pre" lemma="neprísť" pos="SSfs4">nepríde</word>
    <word id="w5" lemma="." pos="Z">.</word>
  </sentence>"""


@pytest.mark.skip()
def test_finding_negation():
    xml = find_negation('Pavol dnes večer nepríde.')
    assert xml == correct_xml


def test_get_pos():
    expected = [
        {'pos': 'SSms1',    'word': 'Pavol'},
        {'pos': 'Dx',       'word': 'dnes'},
        {'pos': 'Dx',       'word': 'večer'},
        {'pos': 'VKdsc-',   'word': 'nepríde'},
        {'pos': 'Z',        'word': '.'}]
    assert get_pos('Pavol dnes večer nepríde.') == expected


def test_get_pos_and_lemma():
    expected = [
        {'pos': 'SSms1',    'word': 'Pavol',    'lemma': 'Pavol'},
        {'pos': 'Dx',       'word': 'dnes',     'lemma': 'dnes'},
        {'pos': 'Dx',       'word': 'večer',    'lemma': 'večer'},
        {'pos': 'VKdsc-',   'word': 'nepríde',  'lemma': 'neprísť'},
        {'pos': 'Z',        'word': '.',        'lemma': '.'}]
    assert get_pos_and_lemma('Pavol dnes večer nepríde.') == expected


def test_create_semi_dataset():
    data = [
        {'pos': 'SSms1',    'word': 'Pavol',    'lemma': 'Pavol'},
        {'pos': 'Dx',       'word': 'dnes',     'lemma': 'dnes'},
        {'pos': 'Dx',       'word': 'večer',    'lemma': 'večer'},
        {'pos': 'VKdsc-',   'word': 'nepríde',  'lemma': 'neprísť'},
        {'pos': 'Z',        'word': '.',        'lemma': '.'},
    ]
    df = create_semi_dataset_sig(data)
    assert len(df) == 9


def test_finding_negator():
    expected = [
        {'pos': 'SSms1',    'word': 'Pavol',    'lemma': 'Pavol',   'negator': 0},
        {'pos': 'Dx',       'word': 'dnes',     'lemma': 'dnes',    'negator': 0},
        {'pos': 'Dx',       'word': 'večer',    'lemma': 'večer',   'negator': 0},
        {'pos': 'VKdsc-',   'word': 'nepríde',  'lemma': 'neprísť', 'negator': 1},
        {'pos': 'Z',        'word': '.',        'lemma': '.',       'negator': 0},
    ]
    results = find_negation('Pavol dnes večer nepríde.')
    assert expected == results
