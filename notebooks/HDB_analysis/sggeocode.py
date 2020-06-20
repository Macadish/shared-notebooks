import urllib
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from pandas.io.json import json_normalize
import geopandas
#import scrapy
import re
import csv
import sys

def getsoup(url, parser="html5lib"):
    req = urllib.request.Request(
        url,
        data=None,
        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) '})
        #'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'})
    with urllib.request.urlopen(req) as response:
       html = response.read()
    soup = BeautifulSoup(html,parser)
    return soup

#with open('./sggeocode.csv', 'w') as f:  # Just use 'w' mode in 3.x
#[(289980,300000),(397806,400000),(489981,500000),(579992,600000),(656291,700000),(799985,800000),(888889,900000),(978201,1000000)]
for postalcode in range(int(sys.argv[1]),int(sys.argv[2])):
    url = 'https://www.streetdirectory.com//api/?mode=search&act=all&output=js&callback=set_data&start=0&limit=1&country=sg&profile=template_1&show_additional=0&no_total=1&q={0:06d}'.format(postalcode)
    soup = getsoup(url)
    try:
        jsontext = soup.find('p').text
    except AttributeError:
        jsontext = soup.find('body').text
    jsontext2 = re.search(r"\[(.*)\]", jsontext).group(1)
    if jsontext2 is not '':
        print(jsontext2+'\t{0:06d}'.format(postalcode))
        #jsondict = json.loads(jsontext2)
        #print('\t'.join([str(i) for i in #list(jsondict.values())]+['{0:06d}'.format(postalcode)]))
        #w = csv.DictWriter(f, jsondict.keys())
        #w.writeheader()
        #w.writerow(jsondict)
    else:
        continue
