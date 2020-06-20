# %%
from __future__ import print_function
%load_ext autoreload
%autoreload 2
from src.scrap import *
from src.load import load_HDB, load_HDB_units
from src.plot import *

import ipywidgets as widgets
from ipywidgets import interact, interactive, interactive_output, fixed, interact_manual
from ipywidgets import Button, HBox, VBox, Layout

from pathlib import Path
import pandas as pd
import re
import numpy as np

# %%
# Load Data
apart_count = load_HDB_units()
apart_count_gb = apart_count.groupby('sc')
hdbdata = load_HDB()


# %%
def filter_HDB(hdbdata, flat_model, flat_type, storey_range, floor_area_sqm, remaining_lease, month):
    bool_array = hdbdata['pc']==hdbdata['pc']
    bool_array = bool_array & hdbdata['flat_model'].isin(flat_model)
    bool_array = bool_array & hdbdata['flat_type'].isin(flat_type)
    bool_array = bool_array & hdbdata['storey_range'].isin(storey_range)
    bool_array = bool_array & (hdbdata['floor_area_sqm'] <= floor_area_sqm[1]) & (hdbdata['floor_area_sqm'] >= floor_area_sqm[0])
    bool_array = bool_array & (hdbdata['month'] <= month[1]) & (hdbdata['month'] >= month[0])
    bool_array = bool_array & (hdbdata['remaining_lease'] <= remaining_lease[1]) & (hdbdata['remaining_lease'] >= remaining_lease[0])
    hdbdata_fil = hdbdata[bool_array].copy()
    return hdbdata_fil


# %%
def basic_buttons(hdbdata):
    """
    Creates a widget box containing basic buttons to filter HDB results
    """
    # https://blog.jupyter.org/introducing-templates-for-jupyter-widget-layouts-f72bcb35a662
    # https://github.com/jupyter-widgets/ipywidgets/issues/1853#issuecomment-349201240
    # Configure layout and set up widgets for plotly
    filterdict = {
    'flat_model': widgets.SelectMultiple(
        options=hdbdata['flat_model'].unique(),
        value=list(hdbdata['flat_model'].unique()),
        description='Flat Model'
        ),
    'flat_type': widgets.SelectMultiple(
        options=hdbdata['flat_type'].unique(),
        value=list(hdbdata['flat_type'].unique()),
        description='Flat Type'
        ),
    'storey_range': widgets.SelectMultiple(
        options=hdbdata['storey_range'].unique(),
        value=list(hdbdata['storey_range'].unique()),
        description='Storeys'
        ),
    'floor_area_sqm': widgets.FloatRangeSlider(
        value=[hdbdata['floor_area_sqm'].min(), hdbdata['floor_area_sqm'].max()],
        min=hdbdata['floor_area_sqm'].min(),
        max=hdbdata['floor_area_sqm'].max(),
        step=1,
        description='Floor Area (sqm):',
        layout=Layout(width='100%')
        ),
    'remaining_lease': widgets.FloatRangeSlider(
        value=[hdbdata['remaining_lease'].min(), hdbdata['remaining_lease'].max()],
        min=hdbdata['remaining_lease'].min(),
        max=hdbdata['remaining_lease'].max(),
        step=1,
        description='Remaining Lease (years):',
        layout=Layout(width='100%')
        ),
    'month': widgets.SelectionRangeSlider(
        options=[(i,i) for i in hdbdata['month'].unique()],
        index=(0, hdbdata['month'].unique().size-1),
        description='Date of transaction',
        layout=Layout(width='100%')
        )
    }
    h1 = HBox([filterdict['flat_model'], filterdict['flat_type'], filterdict['storey_range']])
    h2 = filterdict['floor_area_sqm']
    h3 = filterdict['remaining_lease']
    h4 = filterdict['month']
    filter_widget = VBox([h1,h2,h3,h4])
    return filterdict, filter_widget

# Create button widgets
filterdict, filter_widget = basic_buttons(hdbdata)

# %%
# Filter data
hdbdata_fil = filter_HDB(hdbdata, **{k: v.value for k,v in filterdict.items()})
# Create an output widget. The initial value doesn't matter
hdbdata_fil = filter_HDB(hdbdata, **{k: v.value for k,v in filterdict.items()})
hdbSeries = apart_count_gb[filterdict['flat_type'].value].sum().sum(axis=1)
hdbSeries.name = 'total_units'
initial_fig = plot_sectorcode_values(hdbSeries)
output_plotly = go.FigureWidget(initial_fig, layout=Layout(height='800px', width='100%'))

def update_choropleth(figwid, newfig):
    """
    Update current choropleth with new choropleth!
    """
    with figwid.batch_update():
        for i in newfig.to_dict()['data'][0].keys():
            figwid['data'][0].update({i: newfig.to_dict()['data'][0][i]})
        for i in newfig.to_dict()['layout'].keys():
            figwid['layout'].update({i: newfig.to_dict()['layout'][i]})

# %% Plot
#================ Plot Number of apartments
button_tu = widgets.Button(description="Total Units",
    layout=Layout(width='25%'))

def on_click_tu(b):
    hdbdata_fil = filter_HDB(hdbdata, **{k: v.value for k,v in filterdict.items()})
    hdbSeries = apart_count_gb[filterdict['flat_type'].value].sum().sum(axis=1)
    hdbSeries.name = 'total_units'
    newfig = plot_sectorcode_values(hdbSeries)
    update_choropleth(output_plotly, newfig)

button_tu.on_click(on_click_tu)

#================ Resale transactions
button_sc = widgets.Button(description="Sale Counts",
    layout=Layout(width='25%'))

def on_click_sc(b):
    hdbdata_fil = filter_HDB(hdbdata, **{k: v.value for k,v in filterdict.items()})
    hdb_rec_group = hdbdata_fil.groupby('sc')
    hdbSeries = hdb_rec_group['resale_price'].count()
    newfig = plot_sectorcode_values(hdbSeries)
    update_choropleth(output_plotly, newfig)

button_sc.on_click(on_click_sc)

#================ Number of resale transactions / apartments of the same type
button_scust = widgets.Button(description="Sale Counts / Units of Same Type",
    layout=Layout(width='25%'))

def on_click_scust(b):
    hdbdata_fil = filter_HDB(hdbdata, **{k: v.value for k,v in filterdict.items()})
    hdb_rec_group = hdbdata_fil.groupby('sc')
    salecount = hdb_rec_group['resale_price'].count()
    totcount = apart_count_gb[filterdict['flat_type'].value].sum().sum(axis=1)
    hdbSeries = salecount/totcount
    hdbSeries.name = 'Normalized Sale Counts'
    newfig = plot_sectorcode_values(hdbSeries)
    update_choropleth(output_plotly, newfig)

button_scust.on_click(on_click_scust)

#================ Number of resale transactions / apartments of the all types
button_scuat = widgets.Button(description="Sale Counts / Units of All Type",
    layout=Layout(width='25%'))
hdbdata_fil['flat_type'].unique()
def on_click_scuat(b):
    hdbdata_fil = filter_HDB(hdbdata, **{k: v.value for k,v in filterdict.items()})
    hdb_rec_group = hdbdata_fil.groupby('sc')
    salecount = hdb_rec_group['resale_price'].count()
    totcount = apart_count_gb['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE','MULTI GENERATION'].sum().sum(axis=1)
    hdbSeries = salecount/totcount
    hdbSeries.name = 'Normalized Sale Counts'
    newfig = plot_sectorcode_values(hdbSeries)
    update_choropleth(output_plotly, newfig)

button_scuat.on_click(on_click_scuat)

# %% Display interface
# Create Interface
button_widget = HBox([button_tu, button_sc, button_scust, button_scuat])
ui = VBox([filter_widget, button_widget, output_plotly])
display(ui)
