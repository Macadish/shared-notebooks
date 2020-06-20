# %%
from __future__ import print_function
%load_ext autoreload
%autoreload 2
from src.scrap import *
from src.load import *
from pathlib import Path
import pandas as pd
import re
import numpy as np

# %%
data_raw = Path('/home/jovyan/shared-notebooks/notebooks/HDB_analysis/data/raw/')
data_processed = Path('/home/jovyan/shared-notebooks/notebooks/HDB_analysis/data/processed/')
df = pd.read_csv(data_raw.joinpath('hdb-property-information.csv'))
df = df.rename(columns={"blk_no": "block", 'street':'street_name','1room_sold':'1 ROOM','2room_sold':'2 ROOM','3room_sold':'3 ROOM','4room_sold':'4 ROOM','5room_sold':'5 ROOM','exec_sold':'EXECUTIVE','multigen_sold':'MULTI GENERATION'})

# Convert bldg_contract_town code to the respective town
bct_df = pd.read_csv(data_raw.joinpath('bldg_contract_town.csv'), header=None, names=['code','town'])
bct_dict = bct_df.set_index('code').to_dict()['town']
df['town'] = df['bldg_contract_town'].apply(lambda x:bct_dict[x])
apart_count = df[df['residential'].str.match('Y')].reset_index().copy()

# %%
hdbdata = load_HDB()
hdbdata['remaining_lease']= 99 - (pd.to_datetime(hdbdata['month']).dt.year + pd.to_datetime(hdbdata['month']).dt.month/12 - hdbdata['lease_commence_date'])
hdbdata.loc[hdbdata['flat_type']=='MULTI-GENERATION','flat_type']='MULTI GENERATION'

# %%
addr2pc = load_HDBaddress()
apart_count_gb = apart_count.groupby(['block','street_name','town'])
apart_count['addr'] = pd.Series(list(apart_count_gb.groups.keys()))
apart_count2 = pd.concat([addr2pc.set_index('addr'),apart_count.set_index('addr')], axis=1, sort=False)
unknown_pc = apart_count2.loc[apart_count2['pc'].isna() & ~apart_count2['street_name'].isna()].index

# %% Get pc and x and y
BigBowl = makesoup()

pc = []
x = []
y = []
for ind, i in enumerate(unknown_pc):
    print(ind)
    query = ' '.join(i[0:2])
    try:
        fetched_json = query_onemap(query, BigBowl)
        pc.append(fetched_json['POSTAL'])
        x.append(fetched_json['LONGITUDE'])
        y.append(fetched_json['LATITUDE'])
    except:
        pc.append(np.nan)
        x.append(np.nan)
        y.append(np.nan)

# %%
# After some additional processing
apart_count3 = apart_count2.reset_index().copy()
apart_count3 = apart_count3.drop(columns=['level_0','level_1','level_2','index'])
apart_count3['pc'].isnull()
apart_count3 = apart_count3.dropna(axis=0,subset=['pc'])
apart_count3.to_csv(data_processed.joinpath('hdb-property-information-with-pc.csv'), index=False)

# %%
apart_count4 = pd.read_csv(data_processed.joinpath('hdb-property-information-with-pc.csv'))
