# %% [markdown]
# # Loading HDB resale price data and performing analysis

# %%
%load_ext autoreload
%autoreload 2
import pandas as pd
import scrap
import numpy as np
import csv
from pathlib import Path
import time
import re
import matplotlib.pyplot as plt
import geopandas
import contextily as ctx

# %% [markdown]
# 1. Load files downloaded from [gov.sg](https://data.gov.sg/dataset/resale-flat-prices). Alternatively, use the api provided on the website to download the data.
# 1. Process the HDB data to add postal code, longitude and latitude information.
#     1. To obtain postal code, use `block`, `street_name` and `town` information to pull the postal code from sites like google.com, google map api, streetdirectory.com or onemap.sg. These sites provide convenient apis for large scale queries.
#     1. Since there are roughly 10000 unique HDB blocks compared to 800000+ transactions, it makes more sense to query every unique block than transaction. 
# 1. Pandas loads postal codes stored in .csv files as floats instead of strings. Converting postal codes back to strings is made harder by the presence of NaN.
#
# `hdbdata` includes data that stretches back to 1990 and may include old buildings that no longer exists. In such cases, the postal codes are `NaN`.
#

# %% Load Data
# HDB Data
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
address_dictkey = list(hdbdata.groupby(['block','street_name','town']).groups.keys())

# Previously processed HDB data
hdbdata_processed = pd.read_csv('./data/processed/HDB_202005.csv',sep='\t').drop(columns='Unnamed: 0')
hdbdata_processed.loc[~hdbdata_processed['pc'].isna(),'pc'] = hdbdata_processed.loc[~hdbdata_processed['pc'].isna(),'pc'].apply(lambda x:'{0:06d}'.format(int(x)))

hdbdata_processed

# %% [markdown]
# 1. geocode is a database that links all valid postal codes to their respective address and coordinates. It is obtained using streetdirectory.com's query api.
# 1. addr2pc is a database that links `block`, `street_name` and `town` information to postal codes and coordinates.

# %%
# Postal Code to Address
geocode = pd.read_csv('./data/raw/sggeocode.csv').rename(columns={"Unnamed: 0":"postalcode"})
geocode['postalcode'] = geocode['postalcode'].apply(lambda x:'{0:06d}'.format(x))
geocode = geocode.set_index('postalcode')

# HDB to Postal Code and coordinate
addr2pc = pd.read_csv('data/processed/unique_add2pc.csv',sep='\t').drop(columns='Unnamed: 0')
addr2pc['addr'] = addr2pc['addr'].apply(eval)
addr2pc.loc[~addr2pc['pc'].isna(),'pc'] = addr2pc.loc[~addr2pc['pc'].isna(),'pc'].apply(lambda x:'{0:06d}'.format(int(x)))
#pc2xy = geocode.reindex(list(addr2pc['pc']))[['x','y']]
#addr2pc['x'] = pc2xy.reset_index()['x']
#addr2pc['y'] = pc2xy.reset_index()['y']
addr2pc


# %% [markdown]
# The raw data shows the remaining lease as `x years y months`. To make comparison easier, convert remaining lease to years. Months are converted to decimals.

# %% Convert remaining lease
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

hdbdata_processed['remaining_lease'] = hdbdata_processed['remaining_lease'].apply(convert_lease)

# %% [markdown]
# # Illustrating recent sale count as a choropleth
# We can illustrate the data with a choropleth to visualize the number of recent sales in different areas of Singapore.  Pandas provides an easy way to select data from a range of datetime using conditionals. The data can then be grouped to the areas determined by sector code (first 2 digits of the postal code) using the `groupby()` method. 

# %% Bin by sectorcode
hdbdata_recent = hdbdata_processed[hdbdata_processed['month'] > '2019-01']
hdb_rec_group = hdbdata_recent.groupby(hdbdata_recent['pc'].apply(lambda x:int(x[0:2])))
pc_count = hdb_rec_group['resale_price'].count()
pc_count.name='sale_counts'

# %% [markdown]
# To plot a choropleth, we use Geopandas, a geospatial software that is an extension of pandas. Geopandas accepts most spatial filetypes as input to generate a basic mapping which we can attach relevant data to. In this case, I generated a polygon shapefile using QGIS based on the sector code. Look out a future tutorial on how to generate shapefiles using QGIS!
#
# Next, add the total sale counts from each sector to their respective polygon features. Since geopandas functions similarly to pandas, we can use functions like pd.concat to concatenate the data. Once added, plot the shapefile with the simple `.plot()` command. To add a baselayer to the choropleth, install `contextily`. The package automatically pulls rasterized images from openstreetmap.
#

# %% Plot
shapefile = Path('/home/jovyan/shared-notebooks/notebooks/HDB_analysis/data/processed/sectorcode/sectorcode.shp')
sgshp = geopandas.read_file(shapefile)
sgshp_3857 = sgshp.to_crs(epsg=3857)
sale_counts_shp = pd.concat([sgshp_3857.set_index('sectorcode'),pc_count],axis=1)

plt.close('all')
fig = plt.figure(figsize=(30, 10))
ax = fig.add_subplot()
sale_counts_shp.plot(ax=ax,
                     column='sale_counts',
                     alpha=0.9,
                     edgecolor='k',
                     missing_kwds={'color': 'lightgrey', 'alpha':0.1},
                     legend=True,
                     )
ctx.add_basemap(ax=ax, source=ctx.providers.Stamen.TonerLite)
ax.set_axis_off()
ax.set_title('Number of resale transactions from 2019-01 to 2020-03')

# %% [markdown]
# Recent sales are generally concentrated in areas with many apartments that reached MOP. 
