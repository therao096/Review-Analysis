# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 23:52:59 2020

@author: boston
"""
import os
os.getcwd()
#os.chdir("C:\\Users\\DELL\\P30_Group5_ExcelR_Project")

# Importing Scrapy Library
import scrapy
#from scrapy.crawler import CrawlerProcess
from scrapy.crawler import CrawlerRunner
from twisted.internet import reactor

# Creating a new class to implement Spide
class AmazonReviewsSpider(scrapy.Spider):

 
    # Spider name
    name = 'amazon_reviews'
    custom_settings = {
        'FEED_FORMAT': 'csv',
        'FEED_URI': 'Samsung_SDcardReviews_CURRENT_WEEK.csv'
        }

    # Domain names to scrape
    allowed_domains = ['amazon.in']
 
    # Base URL for the MacBook air reviews
    myBaseUrl   = "https://www.amazon.in/Samsung-MicroSDXC-Memory-Adapter-MB-MC128GA/product-reviews/B06Y63ZKLS/ref=cm_cr_arp_d_paging_btm_next_501?ie=UTF8&reviewerType=all_reviews&pageNumber="
    start_urls=[]
 
    # Creating list of urls to be scraped by appending page number a the end of base url
    for i in range(1,505):
        start_urls.append(myBaseUrl+str(i))
    
    # Defining a Scrapy parser
    def parse(self, response):
            data = response.css('#cm_cr-review_list')
 
            # Collecting product star ratings
            star_rating = data.css('.review-rating')
            review_title = response.css('.a-text-bold span')
            review_date = response.css('.review-date')

            # Collecting user reviews
            comments = data.css('.review-text')
            count = 0
 
            # Combining the results
            for review in star_rating:
                yield{'stars': ''.join(review.xpath('.//text()').extract()),
                      'comment': ''.join(comments[count].xpath(".//text()").extract()),
                      'title': ''.join(review_title[count].css('::text').extract()),
                      'review_date': ''.join(review_date[count].css('::text').extract())
                     }
                count=count+1

# process = CrawlerProcess()
# process.crawl(AmazonReviewsSpider)
# process.start()

runner = CrawlerRunner()

d = runner.crawl(AmazonReviewsSpider)
d.addBoth(lambda _: reactor.stop())
reactor.run() # the script will block here until the crawling is finished