from scrapy.crawler import CrawlerProcess

from korpus_scraper.spiders.korpus import KorpusSpider

process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
})

process.crawl(KorpusSpider)
process.start()
