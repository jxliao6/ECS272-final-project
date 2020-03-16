import dash
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64
import gmap_univ as gmap
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np

def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

algorithm_option = {
    'modularity': ['----N/A----'],
    'Kmeans': [8, 20, 30]
}

app.layout=html.Div([
    html.H1(
            children='Gmaps',
            style={'textAlign': 'center'}
        ),
    
    html.Div([
        html.Label('Dataset'),
        dcc.RadioItems(
            id='dataset_selection',
            options=[
                {'label': 'Universities', 'value': 'Universities'},
                {'label': 'GD Collaboration', 'value': 'GD'},
            ],
            value='Universities'
        ),
        html.Label('Clustering Algorithm'),
        dcc.RadioItems(
            id='clustering_selection',
            options=[
                {'label': 'Modularity', 'value': 'modularity'},
                {'label': 'K-Means', 'value': 'Kmeans'},
            ],
            value='modularity'
        ),
        html.Label('Number of Clusters'),
        dcc.Dropdown(
            id='k_selection'
        ),
    ], style={'columnCount': 3}),
    
    html.Div(
        [
            html.Img(id='gmap_plot', src='')
        ],
        id='gmap_div'
        )
])

@app.callback(
    Output(component_id='k_selection', component_property='options'),
    [Input(component_id='clustering_selection', component_property='value')]
)
def k_selection_option(selected_algorithm):
    if(selected_algorithm == 'modularity'):
        return [{'label': '----N/A----', 'value': '----N/A----'}]
    else:
        return [{'label': 'k = ' + str(i), 'value': i} for i in algorithm_option[selected_algorithm]]

@app.callback(
    Output(component_id='k_selection', component_property='value'),
    [Input(component_id='k_selection', component_property='options')]
)
def k_selection_value(available_options):
    if(len(available_options) == 1):
        return '----N/A----'
    else:
        return 8

@app.callback(
    Output(component_id='gmap_plot', component_property='src'),
    [
        Input(component_id='dataset_selection', component_property='value'),
        Input(component_id='clustering_selection', component_property='value'),
        Input(component_id='k_selection', component_property='value')
    ]
)
def update_gmap(dataset, algorithm, k):
    if(dataset == 'Universities'):
        fig = gmap.solver(algorithm, k)
    out_url = fig_to_uri(fig)
    return out_url

if __name__ == '__main__':
    app.run_server(debug=True)