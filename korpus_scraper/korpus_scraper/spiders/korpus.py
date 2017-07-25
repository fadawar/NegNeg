# -*- coding: utf-8 -*-
import scrapy
from lxml import etree


class Token:
    def __init__(self, idx, word, lemma, pos):
        self.idx = idx
        self.word = word
        self.lemma = lemma
        self.pos = pos

    def __repr__(self, *args, **kwargs):
        return "Token #{} '{}', lemma: '{}', POS tag: '{}'".format(self.idx, self.word, self.lemma, self.pos)

    def to_xml(self):
        root = etree.Element('word', id='w'+str(self.idx), lemma=self.lemma, pos=self.pos)
        root.text = self.word
        return root


class KorpusSpider(scrapy.Spider):
    name = "korpus"
    allowed_domains = ["bonito.korpus.sk"]
    # start_urls = ['https://bonito.korpus.sk/run.cgi/viewattrsx?q=aword%2C[tag%3D%22.*-%22]&q=f&corpname=prim-7.0-public-all&viewmode=sen&refs=%3Ddoc.bogo&fromp=1&setattrs=word&setattrs=lemma&setattrs=tag&allpos=all&setstructs=p&setstructs=g&setrefs=%3Ddoc.bogo&pagesize=10&newctxsize=8']
    start_urls = ['http://bonito.korpus.sk/run.cgi/view?q=atag%2C%5Btag%3D%22S.%2A%22%5D;corpname=prim-7.0-public-all;viewmode=sen;attrs=word%2Clemma%2Ctag&ctxattrs=word%2Clemma%2Ctag&structs=p%2Cg&refs=%3Ddoc.bogo&pagesize=1000&gdexcnt=0&gdexconf=&attr_tooltip=nott;fromp=1']
    http_user = 'gaborik.jozef'
    http_pass = 'doc86rar'

    def parse(self, response):
        next_url = response.css("a#next::attr(href)").extract_first()
        sentences = []
        for sentence in response.css("table#conclines tr td:nth-child(2)"):
            sentences.append(self.parse_sentence(sentence))
        self.save_to_xml(sentences)
        print("Hotovo")

    def parse_sentence(self, sentence):
        tokens = []
        word = None
        idx = 0
        for el in sentence.css('i, span'):
            css_class = el.xpath('@class').extract_first()
            el_name = el.xpath('name()').extract_first()
            if el_name == 'span' and css_class == 'nott':
                word = el.xpath('text()').extract_first()
                if word:
                    word = word.strip()
            elif el_name == 'i' and css_class == 'attr nott':
                try:
                    _, lemma, pos = el.xpath('text()').extract_first().strip().split('/')
                    idx += 1
                    tokens.append(Token(idx, word, lemma, pos))
                except Exception:
                    pass    # when the word is "/"
            elif el_name == 'span' and css_class == 'rtl':
                b1, b2 = el.xpath('b')
                word = b1.xpath('text()').extract_first().strip()
                _, lemma, pos = b2.xpath('text()').extract_first().strip().split('/')
                idx += 1
                tokens.append(Token(idx, word, lemma, pos))
            elif el_name == 'i' and css_class == 'strc nott':
                pass
            else:
                raise Exception('Wrong parsing of sentence')
        return tokens

    def save_to_xml(self, sentences):
        dataset = etree.Element('dataset')
        for sentence in sentences:
            sentence_el = etree.Element('sentence')
            sentence_el.attrib['text'] = ' '.join(token.word for token in sentence)
            for token in sentence:
                sentence_el.append(token.to_xml())
            dataset.append(sentence_el)

        with open('dataset.xml', 'wb') as file:
            file.write(etree.tostring(dataset, encoding='utf8', pretty_print=True))
