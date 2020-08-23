''' 
Introduction:
    The objective of this module is to be used to scrape the respective links for some of the games 
    hosted on Metacritic website ( https://www.metacritic.com/games). The website has a special 
    webpage containing list of all the games, reviews and other metadata. The function featured in 
    this module operates in such a way to navigate through the pages and traverse through the 
    collection of games andd their appropriate metadata.
 

Link :
    https://www.metacritic.com/browse/games/score/metascore/all/all/filtered?page={index}

Parameters:
    page_count: Count be used to control the number of pages to consider for web scraping.

Returns:
    game_links and length of game_links
'''

# This file has the script for navigating through the page indices of page hosting all games
# available on Metacritic website.
# https://www.metacritic.com/browse/games/score/metascore/all/all/filtered?page={index}
# The link above contains list of pages within the range of 0 to 175, denoted with index.
# the index can be used to control the current page to scrape using both the offset 
# and page_count arguments.

import requests as rq
from bs4 import BeautifulSoup
import re
import numpy as np
import pickle

def get_game_links(page_count, offset):
    ## Specifying the root url for the list of games
    BASE_URL = 'https://www.metacritic.com/browse/games/score/metascore/all/all/filtered'
    URLs = [] ## list to hold all the urls to scrape reviews from
    user_agent = {'User-agent': 'Mozilla/5.0'}  # client header (without this beign set, the requeste will not be possible)
    game_links = []
    # Adding the url lists 
    for page_index in np.arange(page_count):
        URLs.append("{}?page={}".format(BASE_URL,page_index+offset))
        # Using the request API to get the page content
        response_object = rq.get(URLs[page_index], headers = user_agent).text
        # Obtain BeautifulSoup DOM tree structure to traversal or manipulation operations
        soup = BeautifulSoup(response_object, 'html.parser')
        # Each page on the consist of 5 to 6 sections.
        # Looping through each page section to obtain link associated with each game
        # finding the page section
        page_sections = soup.find_all("div", {"class": "browse_list_wrapper"})
        # Looping through to get each game...
        for each_page in np.arange(len(page_sections)):
            # detect all links in each section
            game_link = page_sections[each_page].find_all("a", {"class":"title"})
            # Loop through all the links
            for each_page in np.arange(len(game_link)):
                game_links.append(game_link[each_page].get("href"))
    return [game_links, len(game_links)]

# Invoking the function...
# The invokation could be done in a loop though, however doing this would cause 
# all sorts of performance issues on the website. The calls can also be done at a particular interval.
# Besides, for this project the request was made in chuncks and the chunks are pushed to 
# github alongside with the source codes. 
# page_count = 1 and offset=1
LINKS, LINK_SIZE = get_game_links(1,1) 
# By calling the function with the arguments stated above, navigate to  
# https://www.metacritic.com/browse/games/score/metascore/all/all/filtered?page=2
# and perform the implemented operations to scrape the various links that would usedto scrape the
# necessary metadata needed.

# Persist the links to binary file...
## opens file for w as a binary with 'wb' flag.
with open('data_chunck.data', 'wb') as fd:
    ## Storing binary stream using the file handler...
    pickle.dump(LINKS, fd)
