import ipywidgets as widgets
from ipywidgets import interact, interactive, interactive_output, fixed, interact_manual
from ipywidgets import Button, HBox, VBox, Layout
from IPython.display import clear_output
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
import geopandas
import plotly.express as px
import plotly.graph_objs as go

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
