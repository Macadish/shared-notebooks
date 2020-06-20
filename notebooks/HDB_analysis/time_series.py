# %%
from __future__ import print_function
%load_ext autoreload
%autoreload 2
from src.load import load_HDB, load_HDB_units
from src.plot import plot_sectorcode_values
from src.widget import basic_buttons

from jupyter_dash import JupyterDash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import geopandas
import plotly.express as px
import plotly.graph_objs as go
import json

# %%
# Load initial data
hdbdata = load_HDB()
hdbdata['remaining_lease']= 99 - (pd.to_datetime(hdbdata['month']).dt.year + pd.to_datetime(hdbdata['month']).dt.month/12 - hdbdata['lease_commence_date'])
hdbdata.loc[hdbdata['flat_type']=='MULTI-GENERATION','flat_type']='MULTI GENERATION'

# %%
hdbdata_fil = hdbdata.copy()
hdbdata_fil['year'] = pd.to_datetime(hdbdata['month']).dt.year
hdbdata_fil['sc'] = hdbdata_fil['pc'].apply(lambda x:int(x[0:2]))
hdbdata_fil['price_per_sqft_per_year'] = hdbdata_fil['resale_price']/(hdbdata_fil['floor_area_sqm']*10.7639)/hdbdata_fil['remaining_lease']

# Sort by sectorcode and year
hdb_rec_group = hdbdata_fil.groupby(['sc','year'])

# Create traces
df.index.get_level_values(1)
df['price_per_sqft_per_year']
fig = go.Figure()
for idx, df in hdb_rec_group.mean().groupby(level=0):
    fig.add_trace(go.Scatter(x=df.index.get_level_values(1), y=df['price_per_sqft_per_year'],
                    mode='lines',
                    name='SC{}'.format(idx)))

fig.update_layout(title='Price per sqft per year remaining',
                   xaxis_title='Transaction date',
                   yaxis_title='Price ($/sqft/year)')
fig.show()



# %%
