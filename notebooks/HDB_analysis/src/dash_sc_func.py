import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
from pathlib import Path
import geopandas
import json
import plotly.express as px
import plotly.graph_objs as go

def filter_button_dict(hdbdata):
    filterdict = {
        'flat_model': dcc.Checklist(
                        id='flat_model_checklist',
                        options=[{'label': i,'value': i} for i in list(hdbdata['flat_model'].unique())],
                        value=list(hdbdata['flat_model'].unique()),
                        labelStyle={
                            'font-size': '12px',
                            'display': 'inline-block'}),
        'flat_type' : dcc.Checklist(
                        id='flat_type_checklist',
                        options=[{'label': i,'value': i} for i in list(hdbdata['flat_type'].unique())],
                        value=list(hdbdata['flat_type'].unique()),
                        labelStyle={
                            'font-size': '12px',
                            'display': 'inline-block'}),
        'storey_range' : dcc.Checklist(
                        id='storey_range_checklist',
                        options=[{'label': i,'value': i} for i in list(hdbdata['storey_range'].unique())],
                        value=list(hdbdata['storey_range'].unique()),
                        labelStyle={
                            'font-size': '12px',
                            'display': 'inline-block'}),
        'floor_area_sqm' : dcc.RangeSlider(
                        id='floor_area_sqm_range',
                        min=hdbdata['floor_area_sqm'].min(),
                        max=hdbdata['floor_area_sqm'].max(),
                        step=1,
                        value=[70,150],
                        tooltip={'always_visible': True, 'placement': 'top'}),
        'remaining_lease' : dcc.RangeSlider(
                        id='remaining_lease_range',
                        min=hdbdata['remaining_lease'].min(),
                        max=hdbdata['remaining_lease'].max(),
                        step=1,
                        value=[hdbdata['remaining_lease'].min(), hdbdata['remaining_lease'].max()],
                        tooltip={'always_visible': True, 'placement': 'top'}),
        'year' : dcc.RangeSlider(
                        id='year_range',
                        min=hdbdata['year'].min(),
                        max=hdbdata['year'].max(),
                        step=1,
                        value=[2008,hdbdata['year'].max()],
                        tooltip={'always_visible': True, 'placement': 'top'})
    }
    return filterdict


def filter_HDB(hdbdata, flat_model, flat_type, storey_range, floor_area_sqm, remaining_lease, year):
    bool_array = hdbdata['pc']==hdbdata['pc']
    bool_array = bool_array & hdbdata['flat_model'].isin(flat_model)
    bool_array = bool_array & hdbdata['flat_type'].isin(flat_type)
    bool_array = bool_array & hdbdata['storey_range'].isin(storey_range)
    bool_array = bool_array & (hdbdata['floor_area_sqm'] <= floor_area_sqm[1]) & (hdbdata['floor_area_sqm'] >= floor_area_sqm[0])
    bool_array = bool_array & (hdbdata['year'] <= year[1]) & (hdbdata['year'] >= year[0])
    bool_array = bool_array & (hdbdata['remaining_lease'] <= remaining_lease[1]) & (hdbdata['remaining_lease'] >= remaining_lease[0])
    hdbdata_fil = hdbdata[bool_array].copy()
    return hdbdata_fil

def plot_sectorcode_values(hdbSeries, customdata = None, hovertemplate = None):
    """
    Plots hdbSeries, which is a pandas Series with values for each sectorcode
    customdata is a 2D array that contains [custom values] for each sectorcode
    hovertemplate formats the output data for display
    """
    shapefile = Path('/home/jovyan/shared-notebooks/notebooks/HDB_analysis/data/processed/sectorcode/sectorcode.shp')
    sgshp = geopandas.read_file(shapefile)
    geojson = json.loads(sgshp.to_json())
    input_df = hdbSeries.reset_index() #move index to a column
    fig = px.choropleth_mapbox(input_df, geojson=geojson,
                                locations='sc',
                                featureidkey="properties.sectorcode",
                                color=hdbSeries.name,
                                mapbox_style='open-street-map',
                                center = {"lat": 1.3521, "lon": 103.8198},
                                opacity = 0.5,
                                zoom = 10,
                                custom_data = ['sc'],
                                height = 600,
                                color_continuous_scale = 'jet',
                                range_color = list(hdbSeries.quantile([.1, 0.9]))
                                )
    if customdata is not None:
        fig.customdata = customdata
    if hovertemplate is None:
        hovertemplate = "<b>Sector Code: %{customdata[0]}</b><br>"+"{}".format(hdbSeries.name)+": %{z}<extra></extra>"
    fig['data'][0].hovertemplate = hovertemplate
    return fig


def plot_sectorcode_line(hdbSeries_lines, title=None,yaxis_label=None):
    ts_fig = go.Figure()
    for idx, df in hdbSeries_lines.groupby(level=0):
        ts_fig.add_trace(go.Scatter(
                        x=df.index.get_level_values(1),
                        y=df,
                        mode='lines',
                        name='SC{}'.format(idx),
                        customdata=[idx],
                        visible=True))
    if title is None:
        title = ''

    if yaxis_label is None:
        yaxis_label = df.name

    placeholder = ts_fig['layout'].update({'title':title,
                       'xaxis_title':'Transaction date',
                       'yaxis_title':yaxis_label,
                       'height':600})
    return ts_fig

def update_choro(chorofig, sc_list):
    selectedpoints = [ind for ind, i in enumerate(chorofig['data'][0]['locations']) if i in sc_list]
    chorofig['data'][0]['selectedpoints'] = selectedpoints
    return chorofig

def update_ts(tsfig, sc_list):
    for idx, i in enumerate(tsfig['data']):
        if i['customdata'][0] in sc_list:
            tsfig['data'][idx]['visible']=True
        else:
            tsfig['data'][idx]['visible']='legendonly'
    return tsfig

def update_choro_zoom(chorofig, relayoutData):
    try:
        del relayoutData['mapbox._derived'] #mapbox._derived are coordinates derived from the zoom and pan. It should not be specified by the user.
    except KeyError:
        pass
    chorofig['layout'].update(relayoutData)
    return chorofig
