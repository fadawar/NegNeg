import collections

default_values = collections.OrderedDict()
default_values['slovny_druh_substantivum'] = 0
default_values['slovny_druh_adjektivum'] = 0
default_values['slovny_druh_pronominum'] = 0
default_values['slovny_druh_numerale'] = 0
default_values['slovny_druh_verbum'] = 0
default_values['slovny_druh_adverbium'] = 0
default_values['slovny_druh_prepozicia'] = 0
default_values['slovny_druh_konjunkcia'] = 0
default_values['slovny_druh_partikula'] = 0
default_values['slovny_druh_interjekcia'] = 0
default_values['paradigma_substantivna'] = 0
default_values['paradigma_adjektivna'] = 0
default_values['paradigma_zmiesana'] = 0
default_values['paradigma_neuplana'] = 0
default_values['paradigma_zamenna'] = 0
default_values['paradigma_cislovkova'] = 0
default_values['paradigma_prislovkova'] = 0
default_values['rod_muzsky_zivotny'] = 0
default_values['rod_muzsky_nezivotny'] = 0
default_values['rod_zensky'] = 0
default_values['rod_stredny'] = 0
default_values['rod_vseobecny'] = 0
default_values['rod_neurcitelny'] = 0
default_values['cislo_jednotne'] = 0
default_values['cislo_mnozne'] = 0
default_values['pad_1'] = 0
default_values['pad_2'] = 0
default_values['pad_3'] = 0
default_values['pad_4'] = 0
default_values['pad_5'] = 0
default_values['pad_6'] = 0
default_values['pad_7'] = 0
default_values['stupen_pozitiv'] = 0
default_values['stupen_komparativ'] = 0
default_values['stupen_superlativ'] = 0
default_values['aglutinovanost'] = 0
default_values['slovesna_forma_ifninitiv'] = 0
default_values['slovesna_forma_prezent'] = 0
default_values['slovesna_forma_imperativ'] = 0
default_values['slovesna_forma_prechodnik'] = 0
default_values['slovesna_forma_l_ove_pricastie'] = 0
default_values['slovesna_forma_futurum_byt'] = 0
default_values['vid_dokonavy'] = 0
default_values['vid_nedokonavy'] = 0
default_values['vid_obojvidove'] = 0
default_values['osoba_prva'] = 0
default_values['osoba_druha'] = 0
default_values['osoba_tretia'] = 0
default_values['slovna_trieda_participium'] = 0
default_values['slovna_trieda_reflexivum'] = 0
default_values['slovna_trieda_kondicionalova_morfema'] = 0
default_values['slovna_trieda_abreviacia'] = 0
default_values['slovna_trieda_interpunkcia'] = 0
default_values['slovna_trieda_neurcitelny'] = 0
default_values['slovna_trieda_neslovny_element'] = 0
default_values['slovna_trieda_citatovy_vyraz'] = 0
default_values['slovna_trieda_cislica'] = 0
default_values['druh_aktivne'] = 0
default_values['druh_pasivne'] = 0
default_values['forma_vokalizovana'] = 0
default_values['forma_nevokalizovana'] = 0
default_values['kondicialnost'] = 0
default_values['vlastne_meno'] = 0
default_values['chybny_zapis'] = 0


fst = lambda x: x[0]
snd = lambda x: x[1]


def process_pos(pos):
    values = collections.OrderedDict(default_values)
    pos = pos.strip()
    first = fst(pos)

    if first == 'S':
        values['slovny_druh_substantivum'] = 1
        process_substantivum(pos, values)
    elif first == 'A':
        values['slovny_druh_adjektivum'] = 1
        process_adjektivum(pos, values)
    elif first == 'P':
        values['slovny_druh_pronominum'] = 1
        process_pronominum(pos, values)
    elif first == 'N':
        values['slovny_druh_numerale'] = 1
        process_numerale(pos, values)
    elif first == 'V':
        values['slovny_druh_verbum'] = 1
        process_verbum(pos, values)
    elif first == 'G':
        values['slovna_trieda_participium'] = 1
        process_participium(pos, values)
    elif first == 'D':
        values['slovny_druh_adverbium'] = 1
        process_stupen(pos[1], values)
    elif first == 'E':
        try:
            values['slovny_druh_prepozicia'] = 1
            process_forma(pos[1], values)
            process_pad(pos[2], values)
        except AttributeError as e:
            print('Prepozicia POS: ' + pos)
            raise e
    elif first == 'O':
        values['slovny_druh_konjunkcia'] = 1
        if len(pos) == 2 and pos[1] == 'Y':
            values['kondicialnost'] = 1
    elif first == 'T':
        values['slovny_druh_partikula'] = 1
        if len(pos) == 2 and pos[1] == 'Y':
            values['kondicialnost'] = 1
    elif first == 'J':
        values['slovny_druh_interjekcia'] = 1
    elif first == 'R':
        values['slovna_trieda_reflexivum'] = 1
    elif first == 'Y':
        values['slovna_trieda_kondicionalova_morfema'] = 1
    elif first == 'W':
        values['slovna_trieda_abreviacia'] = 1
    elif first == 'Z':
        values['slovna_trieda_interpunkcia'] = 1
    elif first == 'Q':
        values['slovna_trieda_neurcitelny'] = 1
    elif first == '#':
        values['slovna_trieda_neslovny_element'] = 1
    elif first == '%':
        values['slovna_trieda_citatovy_vyraz'] = 1
    elif first == '0':
        values['slovna_trieda_cislica'] = 1
    elif pos == '__None__':
        pass
    else:
        raise AttributeError('Unknown POS tag ' + first)

    if pos.endswith(':r'):
        values['vlastne_meno'] = 1
    if pos.endswith(':q'):
        values['chybny_zapis'] = 1
    return values


def process_substantivum(pos, values):
    process_paradigma(pos[1], values)
    process_rod(pos[2], values)
    process_cislo(pos[3], values)
    process_pad(pos[4], values)


def process_adjektivum(pos, values):
    process_paradigma(pos[1], values)
    process_rod(pos[2], values)
    process_cislo(pos[3], values)
    process_pad(pos[4], values)
    process_stupen(pos[5], values)


def process_pronominum(pos, values):
    try:
        process_paradigma(pos[1], values)
        if len(pos) == 2:
            return
        process_rod(pos[2], values)
        process_cislo(pos[3], values)
        process_pad(pos[4], values)
        if len(pos) >= 6:
            if pos[5] == 'g':
                values['aglutinovanost'] = 1
            else:
                raise AttributeError('Unknown POS - aglutinovanost ' + pos)
    except IndexError as e:
        print('IndexError with POS ' + pos)
        raise e


def process_numerale(pos, values):
    process_paradigma(pos[1], values)
    process_rod(pos[2], values)
    process_cislo(pos[3], values)
    process_pad(pos[4], values)


def process_verbum(pos, values):
    try:
        process_slovesna_forma(pos[1], values)
        process_vid(pos[2], values)
        if pos[1] == 'I':
            return
        process_cislo(pos[3], values)
        process_osoba(pos[4], values)
        if len(pos) >= 5 and pos[5] not in ['+', '-']:
            process_rod(pos[5], values)
    except AttributeError:
        print('AttributeError - Verbum cislo, POS {}'.format(pos))
        raise
    except IndexError:
        print('IndexError - Verbum cislo, POS {}'.format(pos))
        raise


def process_participium(pos, values):
    process_druh(pos[1], values)
    process_rod(pos[2], values)
    process_cislo(pos[3], values)
    process_pad(pos[4], values)
    process_stupen(pos[5], values)


def process_paradigma(paradigma, values):
    if paradigma == 'S':
        values['paradigma_substantivna'] = 1
    elif paradigma == 'A':
        values['paradigma_adjektivna'] = 1
    elif paradigma == 'F':
        values['paradigma_zmiesana'] = 1
    elif paradigma == 'U':
        values['paradigma_neuplana'] = 1
    elif paradigma == 'P':
        values['paradigma_zamenna'] = 1
    elif paradigma == 'D':
        values['paradigma_prislovkova'] = 1
    elif paradigma == 'N':
        values['paradigma_cislovkova'] = 1
    else:
        raise AttributeError('Unknown POS - paradigma')


def process_rod(rod, values):
    if rod == 'm':
        values['rod_muzsky_zivotny'] = 1
    elif rod == 'i':
        values['rod_muzsky_nezivotny'] = 1
    elif rod == 'f':
        values['rod_zensky'] = 1
    elif rod == 'n':
        values['rod_stredny'] = 1
    elif rod == 'h':
        values['rod_vseobecny'] = 1
    elif rod == 'o':
        values['rod_neurcitelny'] = 1
    else:
        raise AttributeError('Unknown POS - rod')


def process_cislo(cislo, values):
    if cislo == 's':
        values['cislo_jednotne'] = 1
    elif cislo == 'p':
        values['cislo_mnozne'] = 1
    else:
        raise AttributeError('Unknown POS - cislo ' + cislo)


def process_pad(pad, values):
    if pad == '1':
        values['pad_1'] = 1
    elif pad == '2':
        values['pad_2'] = 1
    elif pad == '3':
        values['pad_3'] = 1
    elif pad == '4':
        values['pad_4'] = 1
    elif pad == '5':
        values['pad_5'] = 1
    elif pad == '6':
        values['pad_6'] = 1
    elif pad == '7':
        values['pad_7'] = 1
    else:
        raise AttributeError('Unknown POS - pad')


def process_stupen(stupen, values):
    if stupen == 'x':
        values['stupen_pozitiv'] = 1
    elif stupen == 'y':
        values['stupen_komparativ'] = 1
    elif stupen == 'z':
        values['stupen_superlativ'] = 1
    else:
        raise AttributeError('Unknown POS - stupen')


def process_slovesna_forma(forma, values):
    if forma == 'I':
        values['slovesna_forma_ifninitiv'] = 1
    elif forma == 'K':
        values['slovesna_forma_prezent'] = 1
    elif forma == 'M':
        values['slovesna_forma_imperativ'] = 1
    elif forma == 'H':
        values['slovesna_forma_prechodnik'] = 1
    elif forma == 'L':
        values['slovesna_forma_l_ove_pricastie'] = 1
    elif forma == 'B':
        values['slovesna_forma_futurum_byt'] = 1
    else:
        raise AttributeError('Unknown POS - slovesna forma {}' + forma)


def process_vid(vid, values):
    if vid == 'd':
        values['vid_dokonavy'] = 1
    elif vid == 'e':
        values['vid_nedokonavy'] = 1
    elif vid == 'j':
        values['vid_obojvidove'] = 1
    else:
        raise AttributeError('Unknown POS - slovesna forma')


def process_osoba(osoba, values):
    if osoba == 'a':
        values['osoba_prva'] = 1
    elif osoba == 'b':
        values['osoba_druha'] = 1
    elif osoba == 'c':
        values['osoba_tretia'] = 1
    else:
        raise AttributeError('Unknown POS - osoba')


def process_druh(druh, values):
    if druh == 'k':
        values['druh_aktivne'] = 1
    elif druh == 't':
        values['druh_pasivne'] = 1
    else:
        raise AttributeError('Unknown POS - druh')


def process_forma(forma, values):
    if forma == 'v':
        values['forma_vokalizovana'] = 1
    elif forma == 'u':
        values['forma_nevokalizovana'] = 1
    else:
        raise AttributeError('Unknown POS - forma ' + forma)
