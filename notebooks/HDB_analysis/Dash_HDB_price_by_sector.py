# %% [markdown]
# # HDB Price Analysis Dash App
#
# The dash app examines resale prices of HDB flats over the years, segmented by their postal codes. Raw data is obtained from data.gov.sg and onemap.sg.
#
# Users can filter for data based on:
# 1. flat model 
# 1. flat type 
# 1. storey range
# 1. floor area
# 1. years of lease remaining 
# 1. date of transaction. 
# 1. sector code (via graph interaction)
#
# The documentation in this notebook will focus primarily on features in the code.
#
# ## Import
# The app uses several popular libraries to handle data (pandas), geographical data (geopandas), graphs (plotly) and app design (dash). Additionally, it imports utility functions from `src.load` and `src.dash_sc_func`.

# %%
from __future__ import print_function
%load_ext autoreload
%autoreload 2
from src.load import load_HDB, load_HDB_units
from src.dash_sc_func import filter_button_dict, filter_HDB, plot_sectorcode_line, plot_sectorcode_values
from src.dash_sc_func import update_choro, update_ts, update_choro_zoom

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
import re

from functools import wraps

# %% [markdown]
# ## Load Data
# `load_HDB()` loads raw HDB resale data.
# `apart_count` loads numbers of HDB apartments.

# %%
# Load initial HDB data
hdbdata = load_HDB()

# Load HDB apartment count data
apart_count = load_HDB_units()
apart_count_gb = apart_count.groupby('sc')

# %% [markdown]
# ## Load Widgets and Set up Buttons
# Load widgets that useres interact with to filter the HDB sales data. `filter_button_dict()` sets the default widget values based on `hdbdata` and returns a dictionary of `dcc` widgets. The widgets are organized using tables. 
#
# Buttons are used to initiate callbacks and plot different data.

# %%
filterdict = filter_button_dict(hdbdata)
buttons = []
Td1 = lambda x: html.Td(x, style={'padding' : '2px'})
buttonslabel = ['Flat Model','Flat Type','Storey Range','Floor Area Sqm','Remaining Lease','Transaction Year']
for ind, i in enumerate(['flat_model','flat_type','storey_range','floor_area_sqm','remaining_lease','year']):
    buttons.append(html.Tr([Td1(buttonslabel[ind]),Td1(filterdict[i])], style={'padding':'0px'}))

filter_buttons = html.Table(buttons)

# Plot buttons
plot_buttons = html.Div(children=[
html.Button(id='plot_resale_price',
            n_clicks=0, children='Resale Price', style={'display': 'inline-block'}),
html.Button(id='plot_resale_price_sqft',
            n_clicks=0, children='Resale Price per sqft', style={'display': 'inline-block'}),
html.Button(id='plot_resale_price_sqft_yr',
            n_clicks=0, children='Resale Price per sqft per yr', style={'display': 'inline-block'}),
html.Button(id='plot_counts',
            n_clicks=0, children='Transaction Counts',style={'display': 'inline-block'}),
html.Button(id='plot_norm_counts_st',
            n_clicks=0, children='Normalized Counts (by same type)',style={'display': 'inline-block'}),
html.Button(id='plot_norm_counts_at', 
            n_clicks=0, children='Normalized Counts (by all types)',style={'display': 'inline-block'}),
html.Button(id='reset_sc',
        n_clicks=0, children='Reset Sectors', style={'display': 'inline-block'})
])

# %% [markdown]
# ## Initialize Figures
# We first create an initial choropleth and timeseries figure in the app using default values from the widget. 
# `hdbdata` is first filtered using properties except for year of transaction to create a MultiIndexed pandas.Series `hdbSeries`. We use it to plot a time series figure that spans from 1990 to the latest year of transaction. The span for the time series is fixed regardless of what the user selects in the widget.
#
# `hdbSeries` is filtered further based on user's selection of transaction year and aggregated by taking its mean. The value is plotted on a choropleth. 

# %%
# Initialize Time Series
#hdbdata_fil = filter_HDB(hdbdata, **{k: v.value for k,v in filterdict.items()})
hdbdata_fil = filter_HDB(hdbdata, filterdict['flat_model'].value, filterdict['flat_type'].value, filterdict['storey_range'].value, filterdict['floor_area_sqm'].value, filterdict['remaining_lease'].value, [hdbdata['year'].min(),hdbdata['year'].max()])
ts_hdb_rec_group = hdbdata_fil.groupby(['sc','year'])
hdbSeries = ts_hdb_rec_group.mean()['resale_price']
hdbSeries.name = 'resale_price'
tsfig = plot_sectorcode_line(hdbSeries)

# Initialize Choropleth
year = filterdict['year'].value
filtered_hdbSeries = hdbSeries[(hdbSeries.index.get_level_values(1) >= year[0]) & 
                                   (hdbSeries.index.get_level_values(1) <= year[1])]
hdbSeries_choropleth = filtered_hdbSeries.groupby(level=0).mean()
chorofig = plot_sectorcode_values(hdbSeries_choropleth)





# %% [markdown]
# ## Set up callback functions
# We define several functions that are called when their respective buttons are clicked on. The functions processes hdbdata to produce data-of-interetest.
#
# Just like Initialize Figures, `hdbdata` is first filtered by widget selections (except transaction year) to produce `hdbdata_fil`. `hdbdata_fil` is first aggregated by *sector code* and *year* and used to plot the time series. It is then aggregated by transaction years and used to plot the choropleth. 
#
# When working with MultiIndexed data, use groupby(level=x) and get_level_values(x) to manipulate the data based on part of the index. 

# %%
def calc_resale_price(hdbdata_fil, year):
    """
    hdbdata_fil is hbdata filtered by all inputs except for year
    """
    ts_hdb_rec_group = hdbdata_fil.groupby(['sc','year'])
    hdbSeries = ts_hdb_rec_group['resale_price'].mean()
    hdbSeries.name = "Resale Price"
    filtered_hdbSeries = hdbSeries[(hdbSeries.index.get_level_values(1) >= year[0]) & 
                                   (hdbSeries.index.get_level_values(1) <= year[1])]
    hdbSeries_choropleth = filtered_hdbSeries.groupby(level=0).mean()
    return hdbSeries, hdbSeries_choropleth

def calc_resale_price_sqft(hdbdata_fil, year):
    hdbdata_fil['price_per_sqft'] = hdbdata_fil['resale_price']/(hdbdata_fil['floor_area_sqm']*10.7639)
    ts_hdb_rec_group = hdbdata_fil.groupby(['sc','year'])
    hdbSeries = ts_hdb_rec_group['price_per_sqft'].mean()
    hdbSeries.name = '$/Sqft'
    filtered_hdbSeries = hdbSeries[(hdbSeries.index.get_level_values(1) >= year[0]) & 
                                   (hdbSeries.index.get_level_values(1) <= year[1])]
    hdbSeries_choropleth = filtered_hdbSeries.groupby(level=0).mean()
    return hdbSeries, hdbSeries_choropleth

def calc_resale_price_sqft_yr(hdbdata_fil, year):
    hdbdata_fil['price_per_sqft_per_year'] = hdbdata_fil['resale_price']/(hdbdata_fil['floor_area_sqm']*10.7639) / hdbdata_fil['remaining_lease']
    ts_hdb_rec_group = hdbdata_fil.groupby(['sc','year'])
    hdbSeries = ts_hdb_rec_group['price_per_sqft_per_year'].mean()
    hdbSeries.name = "$/Sqft Y"
    filtered_hdbSeries = hdbSeries[(hdbSeries.index.get_level_values(1) >= year[0]) & 
                                   (hdbSeries.index.get_level_values(1) <= year[1])]
    hdbSeries_choropleth = filtered_hdbSeries.groupby(level=0).mean()
    return hdbSeries, hdbSeries_choropleth

def calc_counts(hdbdata_fil, year):
    ts_hdb_rec_group = hdbdata_fil.groupby(['sc','year'])
    hdbSeries = ts_hdb_rec_group.count()['resale_price']
    hdbSeries.name = "Counts"
    filtered_hdbSeries = hdbSeries[(hdbSeries.index.get_level_values(1) >= year[0]) & 
                                   (hdbSeries.index.get_level_values(1) <= year[1])]
    hdbSeries_choropleth = filtered_hdbSeries.groupby(level=0).sum()
    return hdbSeries, hdbSeries_choropleth

def calc_norm_counts_st(hdbdata_fil, year):
    ts_hdb_rec_group = hdbdata_fil.groupby(['sc','year'])
    salecount_groups = ts_hdb_rec_group.count()['resale_price']
    totcount = apart_count_gb[filterdict['flat_type'].value].sum().sum(axis=1)
    tot_salecount_groups = np.array([totcount.loc[i] for i in salecount_groups.index.get_level_values(0)])
    hdbSeries = salecount_groups / tot_salecount_groups
    hdbSeries.name = 'Normalized Counts'
    filtered_hdbSeries = hdbSeries[(hdbSeries.index.get_level_values(1) >= year[0]) & 
                                   (hdbSeries.index.get_level_values(1) <= year[1])]
    hdbSeries_choropleth = filtered_hdbSeries.groupby(level=0).sum()
    return hdbSeries, hdbSeries_choropleth

def calc_norm_counts_at(hdbdata_fil, year):
    ts_hdb_rec_group = hdbdata_fil.groupby(['sc','year'])
    salecount_groups = ts_hdb_rec_group.count()['resale_price']
    totcount = apart_count_gb['total_dwelling_units'].sum()
    tot_salecount_groups = np.array([totcount.loc[i] for i in salecount_groups.index.get_level_values(0)])
    hdbSeries = salecount_groups / tot_salecount_groups
    hdbSeries.name = 'Normalized Counts'
    filtered_hdbSeries = hdbSeries[(hdbSeries.index.get_level_values(1) >= year[0]) & 
                                   (hdbSeries.index.get_level_values(1) <= year[1])]
    hdbSeries_choropleth = filtered_hdbSeries.groupby(level=0).sum()
    return hdbSeries, hdbSeries_choropleth


# %% [markdown]
# ## Initialize Dash App
# The following code is adapted from the Dash tutorial on their main website. 
#
# As an introduction, Dash is based on Plotly, a javascript plotting library that converts json to beautiful html figures. Plotly.py is a python wrapper that converts python code to json, which plotly.js interprets. Dash is an applicaiton tool that lets users customize graphs, widgets and layouts to add interactivity to plots.
#
# Dash creates an app server that hosts the plot. The server is typically found on port 8050, but can be configured by the user. Host is set to 0.0.0.0 to allow access externally or via an ssh tunnel.

# %%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}
app.layout = html.Div(children=[
    html.H1(children='HDB resale data by sector'),
    filter_buttons,
    plot_buttons,
    dcc.Graph(id='choropleth_element', figure=chorofig, style={'width': '60%', 'display': 'inline-block'}),
    dcc.Graph(id='timeseries_element', figure=tsfig, style={'width': '38%','display': 'inline-block'}),  
])
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port='8050')

# %% [markdown]
# ## Configure callback
#
# Callback determines how a figure changes when user input changes. Any function can be designated as a callback function using the decorator `@app.callback`. The argumernts and outputs of the function should match the inputs, states and outputs of the decorator as we discuess below. Note that each output callback can only be linked to a single function, i.e. Dash does not allow multiple callback functions to share the same output. 
#
# `Input` triggers the callback function whenever it detects a change in a target element property. `State` on the other hand 'stores' the changes, but does not trigger the callback. `Output` write the output of the function to a target HTML element. `Output`, `Input` and `State` accepts two arguments, the first is 'id' element, and the second is a property of the element to read or write from. The number of arguments in the callback function should be the sum of `Input` and `State` in the same order, no more no less. The equality must hold even if we don't use the values (such as `'n_clicks'` from buttons) from the inputs. Likewise, the number of values returned by the function should equal the number of `Output` in the callback.
#
#
# For callback functions with multiple inputs, it may be necessary to identify the input that triggered the callback. We can use dash.callback_context to identify the specific input element and property that triggered the most recent callback, and tailor the response accordingly.
#
# In the app, clicking buttons `btn1`-`btn6` triggers a callback that replots the graphs with different sets of data. When users change the filter selections, the changes are stored in `State`. These changes are propagated to the graph only when the buttons are pressed. 
#
# For cross selectivity, the choropleth and timeseries plots are linked such that selecting particular sectors in either graph updates the selection in the other graph. The callback function essentially looks for changes to the selectedData of the choropleth, or changes to restyleData in the time series. Once the changes have been identified, update `'selectedpoints'`  in the choropleth, and `'visible'` in the traces of the time series. Note that `'selectedpoints'` is a set of index that depends on `tsfig[]`
#
# The selection persists between button clicks unless the button `Reset Sectors` is clicked. 
#
# The callback function includes a `State` that tracks the zoom state of the choropleth. Whenever a new figure is generated, its layout is updated to reflect the latest zoom state, hence preserving zoom levels between callbacks. 
#
# The callback function also includes two `States` that store the current figures which can be further processed. 
#
#
# Run `help(app.callback)` to see the relevant docstrings.

# %%
buttondict={
            'output': [Output('choropleth_element', 'figure'),
                Output('timeseries_element', 'figure')],
            'inputs': [Input('plot_resale_price','n_clicks'),
                Input('plot_resale_price_sqft','n_clicks'),
                Input('plot_resale_price_sqft_yr','n_clicks'),
                Input('plot_counts','n_clicks'),
                Input('plot_norm_counts_st','n_clicks'),
                Input('plot_norm_counts_at','n_clicks'),
                Input('reset_sc', 'n_clicks'),
                Input('timeseries_element', 'restyleData'),
                Input('choropleth_element', 'selectedData')],
            'state': [State('flat_model_checklist', 'value'),
                State('flat_type_checklist', 'value'),
                State('storey_range_checklist', 'value'),
                State('floor_area_sqm_range', 'value'),
                State('remaining_lease_range', 'value'),
                State('year_range', 'value'),
                State('choropleth_element', 'relayoutData'),
                State('choropleth_element', 'figure'),
                State('timeseries_element', 'figure')]
            }            
                
@app.callback(**buttondict)
def plot_callback(btn1, btn2, btn3, btn4, btn5, btn6, btn7, tsRestyleData, cSelectedData, flat_model, flat_type, storey_range, 
                  floor_area_sqm, remaining_lease, year, fig_prop, curr_choro_fig, curr_ts_fig):
    # Callback Triggered
    button_id = 'plot_resale_price'
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == '.':
        # If not triggered, return current figures
        return curr_choro_fig, curr_ts_fig
    elif ctx.triggered[0]['prop_id']== 'timeseries_element.restyleData':
        # If the timeseries plot is changed, identify the traces that are still 
        # visible and add that to selectedpoints in the choropleth.
        visible_sc = [i['customdata'][0] for i in curr_ts_fig['data'] if i['visible'] != 'legendonly']
        new_choro_fig = update_choro(curr_choro_fig, visible_sc)
        return new_choro_fig, curr_ts_fig
    elif ctx.triggered[0]['prop_id']=='choropleth_element.selectedData':
        # If the choropleth plot is changed, identify the traces that are still 
        # visible and add that to selectedpoints in the choropleth.
        # The values in selectedpoints should correspond to the index of a sector code
        # in chorofig['data'][0]['locations']
        visible_sc = [i['location'] for i in cSelectedData['points']]
        new_ts_fig = update_ts(curr_ts_fig, visible_sc)
        return curr_choro_fig, new_ts_fig
    elif ctx.triggered[0]['prop_id']=='reset_sc.n_clicks':
        # Sets all sectors to be visible
        visible_sc = [i['customdata'][0] for i in curr_ts_fig['data']]
        new_choro_fig = update_choro(curr_choro_fig, visible_sc)
        new_ts_fig = update_ts(curr_ts_fig, visible_sc)
        return new_choro_fig, new_ts_fig
    elif ctx.triggered[0]['prop_id'].split('.')[1] == "n_clicks":
        # Recalculates data
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # Process data
        hdbdata_fil = filter_HDB(hdbdata, flat_model, flat_type, storey_range, floor_area_sqm, remaining_lease, [hdbdata['year'].min(), hdbdata['year'].max()])
        calc_func = eval('calc'+re.search('^plot(.*)', button_id).group(1))
        hdbSeries, hdbSeries_choropleth = calc_func(hdbdata_fil, year)
        # Plot figures
        tsfig = plot_sectorcode_line(hdbSeries)
        chorofig = plot_sectorcode_values(hdbSeries_choropleth)
        # Preserve visible SC between callbacks
        global vs
        visible_sc = [i['customdata'][0] for i in curr_ts_fig['data'] if i['visible'] != 'legendonly']
        vs = visible_sc
        new_choro_fig = update_choro(chorofig, visible_sc)
        new_ts_fig = update_ts(tsfig, visible_sc)
        return update_choro_zoom(new_choro_fig, fig_prop), new_ts_fig

# %%
