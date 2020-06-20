import pandas as pd
from . import scrap
import numpy as np
import csv
from pathlib import Path
import time
import re
import matplotlib.pyplot as plt
import geopandas
import contextily as ctx

data_raw = Path('/home/jovyan/shared-notebooks/notebooks/HDB_analysis/data/raw/')
data_processed = Path('/home/jovyan/shared-notebooks/notebooks/HDB_analysis/data/processed/')

def load_HDB_raw():
    filename = []
    filename.append('./data/raw/'+'resale-flat-prices-based-on-approval-date-1990-1999.csv')
    filename.append('./data/raw/'+'resale-flat-prices-based-on-approval-date-2000-feb-2012.csv')
    filename.append('./data/raw/'+'resale-flat-prices-based-on-registration-date-from-mar-2012-to-dec-2014.csv')
    filename.append('./data/raw/'+'resale-flat-prices-based-on-registration-date-from-jan-2015-to-dec-2016.csv')
    filename.append('./data/raw/'+'resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv')
    data = []
    for f in filename:
        data.append(pd.read_csv(f))

    hdbdata = pd.concat(data,0, ignore_index=True, sort=True)
    #address_dictkey = list(hdbdata.groupby(['block','street_name','town']).groups.keys())
    return hdbdata

def fetch_gov(url):
    #url = 'https://data.gov.sg/api/action/datastore_search?resource_id=42ff9cfe-abe5-4b54-beda-c88f9bb438ee&limit=10'
    bigbowl = scrap.makesoup()
    bigbowl.setup_proxy(None)
    soup = bigbowl(url)
    jsontext = soup.find('p').text
    jsondict = json.loads(jsontext)
    #https://hackersandslackers.com/json-into-pandas-dataframes/
    data = json_normalize(jsondict,record_path=['result','records'],sep="_")
    return data

def load_HDB(filename=None):
    """
    # Previously processed HDB data
    """
    if filename is None:
        filename = Path('./data/processed/HDB_202005.csv')
    hdbdata = pd.read_csv(filename ,sep='\t').drop(columns='Unnamed: 0')
    # remove entries without postal code
    hdbdata = hdbdata[~hdbdata['pc'].isna()]
    # convert postalcode from int to str
    hdbdata.loc[~hdbdata['pc'].isna(),'pc'] = hdbdata.loc[~hdbdata['pc'].isna(),'pc'].apply(lambda x:'{0:06d}'.format(int(x)))
    # convert remaining lease into years
    hdbdata['transaction_date'] = hdbdata['month']
    hdbdata['year'] = pd.to_datetime(hdbdata['month']).dt.year
    hdbdata['month'] = pd.to_datetime(hdbdata['month']).dt.month
    hdbdata['remaining_lease'] = 99 - (hdbdata['year']+hdbdata['month']/12-hdbdata['lease_commence_date'])
    hdbdata.loc[hdbdata['flat_type']=='MULTI-GENERATION','flat_type']='MULTI GENERATION'
    hdbdata['sc'] = hdbdata['pc'].apply(lambda x:int(x[0:2]))
    return hdbdata

# Available units and postal code
def load_HDB_units(filename=None):
    if filename is None:
        filename = Path(data_processed.joinpath('hdb-property-information-with-pc.csv'))

    apart_count = pd.read_csv(filename)
    apart_count.loc[~apart_count['pc'].isna(),'pc'] = apart_count.loc[~apart_count['pc'].isna(),'pc'].apply(lambda x:'{0:06d}'.format(int(x)))
    apart_count['sc'] = apart_count['pc'].apply(lambda x:int(x[0:2]))
    return apart_count

# Postalcode to coordinate
def load_geocode(filename=None):
    """
    Link address and coordinate information for all valid postal codes
    """
    if filename is None:
        filename = Path('./data/raw/sggeocode.csv')

    geocode = pd.read_csv(filename).rename(columns={"Unnamed: 0":"postalcode"})
    # convert postalcode from int to str
    geocode['postalcode'] = geocode['postalcode'].apply(lambda x:'{0:06d}'.format(x))
    #geocode = geocode.set_index('postalcode')
    return geocode

# HDB to Postal Code and coordinate
def load_HDBaddress(filename=None):
    """
    Loads HDB addresses with the following information:
    postal code, (block, street and town)
    """
    if filename is None:
        Path = 'data/processed/unique_add2pc.csv'

    addr2pc = pd.read_csv('data/processed/unique_add2pc.csv',sep='\t').drop(columns='Unnamed: 0')
    addr2pc['addr'] = addr2pc['addr'].apply(eval) # Converts str to tuple
    addr2pc.loc[~addr2pc['pc'].isna(),'pc'] = addr2pc.loc[~addr2pc['pc'].isna(),'pc'].apply(lambda x:'{0:06d}'.format(int(x)))

    return addr2pc

# Converts available lease. Not useful.
def convert_lease(x):
    if isinstance(x,str):
        years = int(re.search('^([0-9]{1,3})',x).group(1))
        if re.search('month',x):
            months = int(re.search('([0-9]{1,3}) month',x).group(1))
        else:
            # catch monthly
            months = 0
        return years+(months/12.)
    else:
        return x
