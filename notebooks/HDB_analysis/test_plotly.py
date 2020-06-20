# %%
from __future__ import print_function
%load_ext autoreload
%autoreload 2
from src.scrap import *
from src.load import *
import geopandas
import json
import ipywidgets as widgets
from ipywidgets import interact, interactive, interactive_output, fixed, interact_manual
from ipywidgets import Button, HBox, VBox, Layout
from IPython.display import clear_output
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
import plotly.express as px
from urllib.request import urlopen
import plotly.graph_objs as go

# %% [markdown]
# There are several methods to create a choropleth_mapbox in plotly
# * [px.choropleth_mapbox()](https://bit.ly/2YGruPL/)
# * [add_choropleth()](https://bit.ly/2YwNHQ4)
# * px.choropleth()
#
# The object added is a [plotly.graph_objects.Choroplethmapbox](https://bit.ly/37w6pve).

# %% Load Data
hdbdata = load_HDB()
hdbdata['remaining_lease']= 99 - (pd.to_datetime(hdbdata['month']).dt.year - hdbdata['lease_commence_date'])
hdb_rec_group = hdbdata.groupby(hdbdata['pc'].apply(lambda x:int(x[0:2])))
shapefile = Path('/home/jovyan/shared-notebooks/notebooks/HDB_analysis/data/processed/sectorcode/sectorcode.shp')
sgshp = geopandas.read_file(shapefile)
geojson = json.loads(sgshp.to_json())
geojson["features"][0]["properties"]['sectorcode']
input_df = hdb_rec_group.mean().reset_index()
input_df.columns

# %% Adding using plotly express
fig = px.choropleth_mapbox(input_df, geojson=geojson,
                            locations='pc',
                            featureidkey="properties.sectorcode",
                            color="resale_price",
                            mapbox_style="carto-positron",
                            center = {"lat": 1.3521, "lon": 103.8198},
                            opacity = 0.5,
                            zoom = 10,
                            custom_data = ["pc", "floor_area_sqm"]
                            )
# fig['data'][0].customdata
# fig['data'][0].hovertemplate
template = "<b>Sector Code: %{customdata[0]}</b><br>Floor Area: %{customdata[1]} sq m<br>resale_price: $%{z}<extra></extra>"
fig['data'][0].hovertemplate = template
#fig.show("notebook")

fig.to_dict()['layout'].keys()

# %% Adding using add_choropleth specifically for FigureWidget
# https://bit.ly/37w6pve
figwid = go.FigureWidget(fig)
display(figwid)

# %%
fig2 = px.choropleth_mapbox(input_df, geojson=geojson,
                            locations='pc',
                            featureidkey="properties.sectorcode",
                            color="floor_area_sqm",
                            mapbox_style="carto-positron",
                            center = {"lat": 1.3521, "lon": 103.8198},
                            opacity = 0.5,
                            zoom = 10,
                            custom_data = ["pc", "floor_area_sqm"]
                            )
def update_choropleth(figwid, newfig):
    with figwid.batch_update():
        for i in newfig.to_dict()['data'][0].keys():
            figwid['data'][0].update({i: newfig.to_dict()['data'][0][i]})
        for i in newfig.to_dict()['layout'].keys():
            figwid['layout'].update({i: newfig.to_dict()['layout'][i]})

update_choropleth(figwid, fig2)
# %%
# https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.add_choroplethmapbox
figwid = go.FigureWidget()
figwid.add_choroplethmapbox()
