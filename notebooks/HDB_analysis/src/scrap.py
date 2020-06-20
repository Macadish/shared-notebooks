import urllib
import urllib.request
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from pandas.io.json import json_normalize
import geopandas
import scrapy
import re
import requests
import pandas as pd
import numpy as np
from selenium import webdriver
import socket
import time


class makesoup(object):
    def __init__(self, proxy_dict=None):
        # https://stackoverflow.com/questions/3168171/how-can-i-open-a-website-with-urllib-via-proxy-in-python/3168244#3168244
        if proxy_dict is None:
            proxy_dict = get_proxy()
        self.setup_proxy(proxy_dict)

    def __call__(self, url):
        return self.getsoup(url)

    def setup_proxy(self,proxy_dict):
        self.proxy_dict = proxy_dict
        proxy_support = urllib.request.ProxyHandler(proxy_dict)
        opener = urllib.request.build_opener(proxy_support)
        urllib.request.install_opener(opener)

    def getsoup(self, url):
        req = urllib.request.Request(url, data=None,
        headers={'User-Agent': 'Mozilla/5.0 (Macintosh)'})
        attempt, attemptcount, attemptthreshold = True, 0, 1
        while attempt and attemptcount < attemptthreshold:
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    html = response.read()
                attempt = False
            except Exception as e:
                print('Error, trying again.')
                raise
                #self.setup_proxy(get_proxy())
                #attemptcount+=1
                #if attemptcount == attemptthreshold:


        soup = BeautifulSoup(html)
        return soup

def query_sd(query, soupobj, limit=None):
    # Query Street Directory
    if limit is None:
        limit = 1
    query = query.replace(' ','%20')
    url = 'https://www.streetdirectory.com//api/?mode=search&act=all&output=js&callback=set_data&start=0&limit={}&country=sg&profile=template_1&show_additional=0&no_total=1&q={}'.format(limit, query)
    soup = soupobj(url)
    jsontext = soup.find('p').text
    jsontext2 = re.search(r"\[(.*)\]", jsontext).group(1)
    if jsontext2 is not '':
        jsondict = json.loads(jsontext2)
    else:
        raise Exception('Got no search results.')
    return(jsondict)

def query_onemap(query, soupobj):
    time.sleep(0.25)
    # Query OneMap
    query = query.replace(' ','%20')
    if soupobj.proxy_dict is not None:
        soupobj.setup_proxy(None)
    url = 'https://developers.onemap.sg/commonapi/search?searchVal={}&returnGeom=Y&getAddrDetails=Y'.format(query)
    soup = soupobj(url)
    jsontext = soup.find('p').text
    try:
        jsondict = json.loads(jsontext)['results'][0]
    except IndexError as e:
        print('Got no search results.')
        raise
    return(jsondict)

def get_proxy():
    # https://stackoverflow.com/questions/45141407/python-and-selenium-close-all-tabs-without-closing-the-browser
    # https://selenium-python.readthedocs.io/locating-elements.html
    # https://medium.com/@petertc/pro-tips-for-selenium-setup-1855a11f88f8
    opts = webdriver.ChromeOptions()
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--headless')
    with webdriver.Chrome(options=opts) as driver:
        sgproxy='http://spys.one/free-proxy-list/SG/'
        driver.get(sgproxy)
        #selenium.common.exceptions.WebDriverException:
        # https://www.guru99.com/handling-dynamic-selenium-webdriver.html
        ip = pd.Series([i.text for i in driver.find_elements_by_xpath('/html/body/table[2]/tbody/tr[5]/td/table/tbody/tr/td[1]')][3:23], name='ip')
        proxy_type = pd.Series([i.text for i in driver.find_elements_by_xpath('/html/body/table[2]/tbody/tr[5]/td/table/tbody/tr/td[2]')][3:23], name='proxy_type')
        latency = pd.Series([i.text for i in driver.find_elements_by_xpath('/html/body/table[2]/tbody/tr[5]/td/table/tbody/tr/td[6]')][1:21], name='latency', dtype='float32')
        tab = pd.concat([ip,proxy_type,latency],axis=1)
        ind_true = tab[tab['proxy_type']=='HTTP']['latency'].idxmin()
        https_ip = tab.iloc[ind_true]['ip']
        http_ip = tab.iloc[ind_true]['ip']
        proxy_dict = {'https':https_ip,'http':http_ip}
    return proxy_dict
