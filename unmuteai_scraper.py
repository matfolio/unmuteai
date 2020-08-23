import requests as rq
from bs4 import BeautifulSoup
import re
import numpy as np
import pickle

# file name (csv)
# Creating csv file for scraped data: structured format for the game reviews and other features...
review_data = "data_chunk_1.csv"
## headers
headers = " , ".join(["title", "platform", "release_date", "developer", "genres", "reviewer", "metascore", "review"])
headers = headers+"\n"
# open a file with 'w' flag
file = open(review_data, "w", encoding="utf-8")
## Save headers to the file
file.write(headers)

# Read binary streams from saved links in the 'unmuteai_link_scraper.py'
with open('data_chunk.data', 'rb') as fd:
    # read the data as binary data stream
    LINKS = pickle.load(fd)

def get_game_details(links, review_body_index):
    BASE_URL = "https://www.metacritic.com"
    URLs = [] ## list to hold all the urls to scrape reviews from
    # client header (without this being set, the requeste will request)
    user_agent = {'User-agent': 'Mozilla/5.0'}
    count = 0
    for index, PATH in enumerate(links):
        # initializing relative paths...
        URLs.append("{}{}".format(BASE_URL,PATH))
        # Using the request API to get the page content
        # The result is a respons object.
        response_object = rq.get(URLs[index], headers = user_agent).text
        # Obtain BeautifulSoup DOM tree structure to traversal or manipulation operations
        soup = BeautifulSoup(response_object, 'html.parser')
        # Scraping review body...
        reviews_check = soup.select(".reviews.critic_reviews")
        ## Checks if review is empty.
        if len(reviews_check) > 0:
            ## gets the review title...
            title = soup.select_one(".hover_none")
            if title:
                title = title.text.strip().replace(",","")
            
            ## gets platform on which the game is compatible with...
            platform = soup.select_one(".product_title .platform")
            if platform:
                platform = platform.text.strip()
            
            ## gets game release date...
            release_date = soup.select_one(".summary_detail.release_data .data")
            if release_date:
                release_date = release_date.text.strip().replace(",","")
                release_date = re.sub("\s+","-",release_date)
            #metascore = soup.select_one(".metascore_w.xlarge.game.positive span").next_element.lstrip()
            ## gets the game developer(s)...
            developer = soup.select_one(".summary_detail.developer .data")
            if developer:
                developer = developer.text.strip()
                developer = developer.replace(","," |")
            
            ## game genre...
            genres = soup.select(".summary_detail.product_genre .data")
            genre_list = []
            for item in genres:
                if item:
                    genre_list.append(item.text) 
            genres = " | ".join(genre_list)
            ## gets reviews
            review_section = soup.find_all("li", {"class":"critic_review"})
            if review_section[review_body_index]:
                ## gets reviewer organization...
                reviewer = review_section[review_body_index].select_one(".review_stats .review_critic .source")
                if reviewer:
                    reviewer = reviewer.text.strip()
                ## gets metascore...
                metascore = review_section[review_body_index].select_one(".review_stats .review_grade .metascore_w").text.strip()
                ## gets review body...
                review = review_section[review_body_index].select_one(".review_body").text.strip()
                if review:
                    review = review.replace(",","")
                    review = re.sub("\s+", " ",review)
                    review = re.sub("\"|\'|\“|\”","",review)
                #print(developer)
            else:
                # select the first review body of the review section....
                if review_section[0]:
                    ## gets reviewer organization...
                    reviewer = review_section[0].select_one(".review_stats .review_critic .source")
                if reviewer:
                    reviewer = reviewer.text.strip()
                ## gets metascore...
                metascore = review_section[0].select_one(".review_stats .review_grade .metascore_w").text.strip()
                ## gets reviews
                review = review_section[0].select_one(".review_body").text.strip()
                if review:
                    review = review.replace(",","")
                    review = re.sub("\s+", " ",review)
                    review = re.sub("\"|\'|\“|\”","",review)
                #print(developer)
            rows = "{},{},{},{},{},{},{},{}".format(title,platform,release_date,developer,genres,reviewer,metascore,review+'\n') 
            file.write(rows)
            #count = count + 1
            #print(metascore," - ",count)
            #print()
            #print(review)


# Each critic review section many review sub-sections.
# each sub-section contains: metascore, reviewer, review body.
# invoke the function with the list of links collected from unmuteai_link_scraper.py.
# Invoking the function with the second reviewer...
review_body_index = 1
get_game_details(LINKS,review_body_index)
file.close()


