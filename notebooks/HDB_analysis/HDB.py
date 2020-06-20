# %%
%load_ext autoreload
%autoreload 2
import urllib
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from pandas.io.json import json_normalize
import geopandas
import scrapy
import re
from src import scrap
from src import load


# https://pypi.org/project/pyshp/
# https://en.wikipedia.org/wiki/Shapefile

# https://docs.ckan.org/en/2.8/api/
# https://docs.ckan.org/en/ckan-2.7.0/maintaining/datastore.html#making-a-datastore-api-request
# https://docs.ckan.org/en/ckan-2.7.0/maintaining/datastore.html#ckanext.datastore.logic.action.datastore_search
# https://data.gov.sg/dataset/ckan-datastore-search

def getsoup(url):
    req = urllib.request.Request(
    url,
    data=None,
    headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'})

    with urllib.request.urlopen(req) as response:
       html = response.read()

    soup = BeautifulSoup(html)
    return soup



# %%
url = 'https://data.gov.sg/api/action/datastore_search?resource_id=42ff9cfe-abe5-4b54-beda-c88f9bb438ee&limit=10'
bigbowl = scrap.makesoup()
bigbowl.setup_proxy(None)
soup = bigbowl(url)
jsontext = soup.find('p').text
jsondict = json.loads(jsontext)
#https://hackersandslackers.com/json-into-pandas-dataframes/
data = json_normalize(jsondict,record_path=['result','records'],sep="_")
data



# %% [markdown]
# https://developers.google.com/maps/documentation/geocoding/start
#
# 1. Address to geolocation
# 1. Distance from MRT
# 1. Time to Buona Vista
# 1. Time to CBD
# 1. Factors affecting prices (Housing features)
# 1. Factors affecting prices (Inflation, interest rates, wage, population growth, birth rate vs immigration, economic outlook, etc.)
# 1. Finding under valued properties (with a map API)
# 1. Looking for pricing trends in different areas
# 1. Countour maps of how long it takes to get from an area to a destination (cbd, buonavista, groceries, parent's home)
# 1.
# 1.
# 1.
# 1.
#
# # Geocoding

# %%
postalcode = 308218
url2 = 'https://www.streetdirectory.com//api/?mode=search&act=all&output=js&callback=set_data&start=0&limit=1&country=sg&profile=template_1&show_additional=0&no_total=1&q={}'.format(postalcode)
soup = getsoup(url2)
jsontext = soup.find('p').text
jsontext2 = re.search(r"\[(.*)\]", jsontext).group(1)
if jsontext2 is not '':
    jsondict = json.loads(jsontext2)
else:
    break
jsondict



# %%
100000/60/60

jsondict['x']
