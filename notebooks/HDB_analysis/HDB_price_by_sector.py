# %%
from __future__ import print_function
%load_ext autoreload
%autoreload 2
from src.scrap import *
from src.load import *
import ipywidgets as widgets
from ipywidgets import interact, interactive, interactive_output, fixed, interact_manual
from ipywidgets import Button, HBox, VBox, Layout
from IPython.display import clear_output
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
import geopandas
import plotly.express as px
import plotly.graph_objs as go

# %% [markdown]
# * Find average price of each sector in 2019
# * Find average price of each sector for certain floor
# * Find average counts per sector

# %%
# Load initial data
hdbdata = load_HDB()


# %%
# Set up Widget to filter data
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
'year': widgets.SelectionRangeSlider(
    options=[(i,i) for i in hdbdata['year'].unique()],
    index=(0, hdbdata['year'].unique().size-1),
    description='Year of transaction',
    layout=Layout(width='100%')
    )
}

# https://blog.jupyter.org/introducing-templates-for-jupyter-widget-layouts-f72bcb35a662
# https://github.com/jupyter-widgets/ipywidgets/issues/1853#issuecomment-349201240
# Configure layout and set up widgets for plotly
h1 = HBox([filterdict['flat_model'], filterdict['flat_type'], filterdict['storey_range']])
h2 = filterdict['floor_area_sqm']
h3 = filterdict['remaining_lease']
h4 = filterdict['year']
filter_widget = VBox([h1,h2,h3,h4])

# Filter data
def filter_HDB(flat_model, flat_type, storey_range, floor_area_sqm, remaining_lease, year):
    bool_array = hdbdata['pc']==hdbdata['pc']
    bool_array = bool_array & hdbdata['flat_model'].isin(flat_model)
    bool_array = bool_array & hdbdata['flat_type'].isin(flat_type)
    bool_array = bool_array & hdbdata['storey_range'].isin(storey_range)
    bool_array = bool_array & (hdbdata['floor_area_sqm'] <= floor_area_sqm[1]) & (hdbdata['floor_area_sqm'] >= floor_area_sqm[0])
    bool_array = bool_array & (hdbdata['year'] <= year[1]) & (hdbdata['year'] >= year[0])
    bool_array = bool_array & (hdbdata['remaining_lease'] <= remaining_lease[1]) & (hdbdata['remaining_lease'] >= remaining_lease[0])
    hdbdata_fil = hdbdata[bool_array].copy()
    return hdbdata_fil

# %%
def plot_plotly(hdbSeries):
    """
    Plots data grouped by sector code
    """
    shapefile = Path('/home/jovyan/shared-notebooks/notebooks/HDB_analysis/data/processed/sectorcode/sectorcode.shp')
    sgshp = geopandas.read_file(shapefile)
    geojson = json.loads(sgshp.to_json())
    input_df = hdbSeries.reset_index() #move index to a column
    fig = px.choropleth_mapbox(input_df, geojson=geojson,
                                locations='pc',
                                featureidkey="properties.sectorcode",
                                color="value",
                                mapbox_style='open-street-map',
                                center = {"lat": 1.3521, "lon": 103.8198},
                                opacity = 0.5,
                                zoom = 10,
                                custom_data = ["pc"],
                                height = 800,
                                color_continuous_scale = 'jet'
                                )
    template = "<b>Sector Code: %{customdata[0]}</b><br>Value: $%{z}<extra></extra>"
    fig['data'][0].hovertemplate = template
    return fig

# Initial Figure
hdbdata_fil = filter_HDB(**{k: v.value for k,v in filterdict.items()})
hdb_rec_group = hdbdata_fil.groupby('sc')
hdbSeries = hdb_rec_group['resale_price'].mean()
hdbSeries.name = "value"
initial_fig = plot_plotly(hdbSeries)
output_plotly = go.FigureWidget(initial_fig, layout=Layout(height='800px', width='100%'))



# %% Create plot buttons
def plot_function(fnc):
    hdbdata_fil = filter_HDB(**{k: v.value for k,v in filterdict.items()})
    hdbSeries = fnc(hdbdata_fil)
    fig = plot_plotly(hdbSeries)
    return fig

def update_choropleth(figwid, newfig):
    """
    Update current choropleth with new choropleth!
    """
    with figwid.batch_update():
        for i in newfig.to_dict()['data'][0].keys():
            figwid['data'][0].update({i: newfig.to_dict()['data'][0][i]})
        for i in newfig.to_dict()['layout'].keys():
            figwid['layout'].update({i: newfig.to_dict()['layout'][i]})

#================ Plot Mean Resale Price
def resale_price_mean(hdbdata_fil):
    hdb_rec_group = hdbdata_fil.groupby('sc')
    hdbSeries = hdb_rec_group['resale_price'].mean()
    hdbSeries.name = "value"
    return hdbSeries

button_rp = widgets.Button(description="Resale Price",
    layout=Layout(width='25%'))
def on_click_rp(b):
    newfig = plot_function(resale_price_mean)
    update_choropleth(output_plotly, newfig)

button_rp.on_click(on_click_rp)

#================= Plot Mean Resale Price per sqft
def price_per_sqft_mean(hdbdata_fil):
    hdbdata_fil['price_per_sqft'] = hdbdata_fil['resale_price']/(hdbdata_fil['floor_area_sqm']*10.7639)
    hdb_rec_group = hdbdata_fil.groupby('sc')
    hdbSeries = hdb_rec_group['price_per_sqft'].mean()
    hdbSeries.name = "value"
    return hdbSeries

button_rpps = widgets.Button(description="Resale Price per sqft",
    layout=Layout(width='25%'))
def on_click_rpps(b):
    newfig = plot_function(price_per_sqft_mean)
    update_choropleth(output_plotly, newfig)

button_rpps.on_click(on_click_rpps)

#================ Plot Mean Resale Price per sqft per year
def price_per_sqft_mean_per_remaining_lease(hdbdata_fil):
    hdb_rec_group = hdbdata_fil.groupby('sc')
    hdbdata_fil['price_per_sqft_per_year'] = hdbdata_fil['resale_price']/(hdbdata_fil['floor_area_sqm']*10.7639)/hdbdata_fil['remaining_lease']
    hdbSeries = hdb_rec_group['price_per_sqft_per_year'].mean()
    hdbSeries.name = "value"
    return hdbSeries

button_rppspy = widgets.Button(description="Resale Price per sqft per year",
    layout=Layout(width='25%'))
def on_click_rppspy(b):
    newfig = plot_function(price_per_sqft_mean_per_remaining_lease)
    update_choropleth(output_plotly, newfig)

button_rppspy.on_click(on_click_rppspy)

#================= Plot Sale Count
def sale_counts(hdbdata_fil):
    hdb_rec_group = hdbdata_fil.groupby('sc')
    hdbSeries = hdb_rec_group['resale_price'].count()
    hdbSeries.name = "value"
    return hdbSeries

button_sc = widgets.Button(description="Sale Count",
    layout=Layout(width='25%'))
def on_click_sc(b):
    newfig = plot_function(sale_counts)
    update_choropleth(output_plotly, newfig)

button_sc.on_click(on_click_sc)

# %% Display interface
# Create Interface
button_widget = HBox([button_rp, button_rpps, button_rppspy, button_sc])
ui = VBox([filter_widget, button_widget, output_plotly])
display(ui)


# %%
# Matplotlib
def plot_mpl(hdbSeries):
    shapefile = Path('/home/jovyan/shared-notebooks/notebooks/HDB_analysis/data/processed/sectorcode/sectorcode.shp')
    sgshp = geopandas.read_file(shapefile)
    sgshp_3857 = sgshp.to_crs(epsg=3857)
    data_shp = pd.concat([sgshp_3857.set_index('sectorcode'),hdbSeries],axis=1)
    plt.close('all')
    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot()
    data_shp.plot(ax=ax,
                         column='resale_price',
                         alpha=0.9,
                         edgecolor='k',
                         missing_kwds={'color': 'lightgrey', 'alpha':0.1},
                         legend=True,
                         )
    ctx.add_basemap(ax=ax, source=ctx.providers.Stamen.TonerLite)
    ax.set_axis_off()
    ax.set_title('Average price')

# Configure layout and set up widgets for matplotlib
output_mpl = widgets.Output()
button_mpl = widgets.Button(description="Plot matplotlib")
def on_mpl_clicked(b):
    with output_mpl:
        clear_output(wait=True)
        hdbdata_fil = filter_HDB(**{k: v.value for k,v in filterdict.items()})
        fnc_out = resale_price_mean(hdbdata_fil)
        plot_mpl(fnc_out)
        show_inline_matplotlib_plots()

ui2 = VBox([filter_widget, button_mpl, output_mpl])
button_mpl.on_click(on_mpl_clicked) # Define button callback
display(ui2)
# %%
