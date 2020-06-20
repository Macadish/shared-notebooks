import geopandas
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from pathlib import Path
import json

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
                                custom_data = ["sc"],
                                color_continuous_scale = 'jet',
                                range_color = list(hdbSeries.quantile([.1, 0.9]))
                                )
    if customdata is not None:
        fig.customdata = customdata
    if hovertemplate is None:
        hovertemplate = "<b>Sector Code: %{customdata[0]}</b><br>"+"{}".format(hdbSeries.name)+": %{z}<extra></extra>"
    fig['data'][0].hovertemplate = hovertemplate
    return fig


def plot_sectorcode_line(HDBgb,title=None,yaxis_label=None):
    ts_fig = go.Figure()
    for idx, df in HDBgb.mean().groupby(level=0):
        ts_fig.add_trace(go.Scatter(x=df.index.get_level_values(1), y=df['resale_price'],
                        mode='lines',
                        name='SC{}'.format(idx)))
    if title is None:
        title = ''
        
    if yaxis_label is None:
        yaxis_label = df.name
        
    placeholder = ts_fig['layout'].update({'title':title,
                       'xaxis_title':'Transaction date',
                       'yaxis_title':yaxis_label})
    return ts_fig
