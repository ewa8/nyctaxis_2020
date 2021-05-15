from os import stat_result
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import time
import pickle
import json
import sklearn

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
token = open(".mapbox_token").read()



#methods for converting the unix time to dates
def unixTimeMillis(dt):
    ''' Convert datetime to unix timestamp '''
    return int(time.mktime(dt.timetuple()))

def unixToDatetime(unix):
    ''' Convert unix timestamp to datetime. '''
    return pd.to_datetime(unix,unit='s')

#gets marks for the slider
def getMarks(dates):
    ''' Returns the marks for labeling. 
        Every Nth value will be used.
    '''
    result = {}
    for date in dates:
        result[unixTimeMillis(date)] = str(date.strftime('%m-%d'))

    return result


#loading the geojson data
file = open("NYC_Taxi_Zones.geojson", "r")
geojson = file.read()
gj = json.loads(geojson)

#reading in the data
data = pd.read_pickle('half_data.pkl')
data['count'] = 1
data = data[data['PULocationID'] <= 263]
yellow_data = data[data['type'] == 'yellow']
green_data = data[data['type'] == 'green']

#reading in machine learning map data
ml_data = pd.read_pickle('ml_data.pkl')

#a dictionary linking zone ids to zone names
zones_dict = {location['properties']['location_id']: location['properties']['zone'] for location in gj['features']}


#create a daterange and NYC closure events
daterange = pd.date_range(start='2020',end='2021',freq='D')
dates = {'Dates': ['2020-03-07',  '2020-03-16', '2020-03-22', '2020-06-08', '2020-07-06', '2020-07-19', '2020-09-09', '2020-09-29',
    '2020-11-19'], 'Desc': ['NY Governor Andrew Cuomo declares a state of emergency', 'NYC public schools close',
    'NYS on Pause Program begins, all non-essential workers must stay home', 'NYC begins Phase 1 reopening',
    'NYC begins Phase 3 of reopening, without indoor dining', 'NYC begins Phase 4 reopening, excluding malls, museums and indoor dining/bars', 'Malls in NYC reopen at 50% capacity with no indoor dining. Casinos reopen across NYS at 25%% capacity.', 'Elementary students return to public school classrooms across NYC', 'NYC schools switch to all-remote']}
dates_df = pd.DataFrame(dates)
dates_df['Dates'] = pd.to_datetime(dates_df['Dates'])
min_date = daterange.min() + pd.to_timedelta(1, unit='D')

#create a dictionary for the titles
title_dict = {'pickup_date': 'Pick-up date', 'dropoff_date': 'Drop-off date', 'total': 'Pick-ups and drop-offs',
 'Trip_distance': 'Trip distance', 'Fare_amount': 'Fare amount', 'Tip_amount': 'Tip amount', 'Passenger_count': 'Passenger count'}

#create colorscales
colors_sc = {'yellow': px.colors.sequential.YlOrBr, 'green': px.colors.sequential.algae, 'total':px.colors.sequential.Sunsetdark}


#loading the machine learning model
loaded_model = pickle.load(open('best_model_2.pkl', 'rb'))

#zone look-ups for getting the location id for each zone
zone_lookup = {feature['properties']['location_id']: feature for feature in gj['features']}
selections = set()
selections_copy = set()

#gets highlighted zones from the machine learning choropleth
def get_highlights(selections, geojson=gj, zone_lookup=zone_lookup):
    geojson_highlights = dict()
    for k in geojson.keys():
        if k != 'features':
            geojson_highlights[k] = gj[k]
        else:
            geojson_highlights[k] = [zone_lookup[str(selection)] for selection in selections]   
    return geojson_highlights


#colors for machine learning results
colors_res = {'1': 'Green ', '0': 'Yellow'}


app.layout = html.Div([
    html.Div([
        html.H1(children='New York City Taxi Patterns During Covid-19')
    ], style={"marginLeft": 10, "marginTop":15, "marginBottom":15, 'color':'#620042'}), #, 'color':'#7f1d70'#ea656f
    dcc.Tabs([
        dcc.Tab(label='NYC Map', children=[
            html.Div([
            dbc.Row([
                dbc.Col(
                    dcc.RadioItems(
                        id='colors',
                        options=[
                            {'label': 'Green', 'value': 'green'},
                            {'label': 'Yellow', 'value': 'yellow'},
                            {'label': 'Total', 'value': 'total'}
                        ],
                        value='total',
                        labelStyle={'display': 'block'}
                    ), width='auto'
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id='display_selection',
                        options=[
                            {'label': 'Pick-ups', 'value': 'pickup_date'},
                            {'label': 'Drop-offs', 'value': 'dropoff_date'},
                            {'label': 'Pick-ups & Drop-offs', 'value': 'total'},
                            {'label': 'Average distance', 'value': 'Trip_distance'},
                            {'label': 'Average fare amount', 'value': 'Fare_amount'},
                            {'label': 'Average tips', 'value': 'Tip_amount'},
                            {'label': 'Average no. of passengers', 'value': 'Passenger_count'}
                        ],
                        value='pickup_date',
                        clearable=False,
                    )
                , width=3),

            ])], style={'marginLeft': 20, 'marginTop': 20}
            ),
            html.Div([
                dcc.Slider(
                    id='year_slider',
                    min = unixTimeMillis(min_date),
                    max = unixTimeMillis(daterange.max()),
                    value = unixTimeMillis(min_date),
                    marks=getMarks(dates_df[dates_df['Dates'] != '2020-03-16']['Dates']),
                    updatemode='drag',
                    step=604800
                )
            ], style={"width": "90%"}
            ),
            html.Div([
                dcc.Markdown(id='slider-output-container')
            ], style={'marginLeft': 20}
            ),
            html.Div([
                dcc.Loading(
                    dcc.Graph(
                        id='choropleth',
                        figure=dict(
                            data=[],
                            layout={},
                        ),
                    )
                )
            ], style={'marginBottom': 20}
            )
        ]),
        dcc.Tab(label='Plots', children=[
            dcc.Tabs([
                dcc.Tab(label='Histograms', children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col(
                                dcc.RadioItems(
                                    id='colors_2',
                                    options=[
                                        {'label': 'Green', 'value': 'green'},
                                        {'label': 'Yellow', 'value': 'yellow'},
                                        {'label': 'Total', 'value': 'total'}
                                    ],
                                    value='green',
                                    labelStyle={'display': 'block'}
                                ), width='auto'
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='display_2',
                                    options=[
                                        {'label': 'Pick-ups', 'value': 'pickup_date'},
                                        {'label': 'Drop-offs', 'value': 'dropoff_date'},
                                        {'label': 'Total distance', 'value': 'Trip_distance'},
                                        {'label': 'Total fare amount', 'value': 'Fare_amount'},
                                        {'label': 'Total tip amount', 'value': 'Tip_amount'},
                                        {'label': 'No. of passengers', 'value': 'Passenger_count'}
                                    ],
                                    value='pickup_date',
                                    clearable=False,
                                )
                            , width=3),

                        ]), 
                        dcc.Loading(
                            dcc.Graph(
                                id='histogram'
                            )
                        )
                    ], style={'marginLeft': 20, 'marginTop': 20, 'marginBottom':10, 'marginRight': 20}
                    )
                ]),
                dcc.Tab(label='Daily patterns', children=[  
                    html.Div([
                        dbc.Row([
                            dbc.Col(
                                dcc.RadioItems(
                                    id='colors_21',
                                    options=[
                                        {'label': 'Green', 'value': 'green'},
                                        {'label': 'Yellow', 'value': 'yellow'},
                                        {'label': 'Total', 'value': 'total'}
                                    ],
                                    value='green',
                                    labelStyle={'display': 'block'}
                                ), width='auto'
                            ), 
                            dbc.Col(
                                dcc.Dropdown(
                                    id='display_21',
                                    options=[
                                        {'label': 'Total distance', 'value': 'Trip_distance'},
                                        {'label': 'Average distance', 'value': 'avg_Trip_distance'},
                                        {'label': 'Fare amount', 'value': 'Fare_amount'},
                                        {'label': 'Average fare amount', 'value': 'avg_Fare_amount'},
                                        {'label': 'Total tips', 'value': 'Tip_amount'},
                                        {'label': 'Average tips', 'value': 'avg_Tip_amount'},
                                        {'label': 'No. of passengers', 'value': 'Passenger_count'},
                                        {'label': 'Average no. of passengers', 'value': 'avg_Passenger_count'}
                                    ],
                                    value='Trip_distance',
                                    clearable=False,
                                )
                            , width=3),

                        ]),
                        dcc.Loading(
                            dcc.Graph(
                                id='line_plot'
                            )
                        )
                        ], style={'marginLeft': 20, 'marginTop': 20, 'marginBottom':10, 'marginRight':20}
                    )
                ])
            ], colors={
        "border": "white",
        "primary": "#ea656f",
        "background": "#fff5e6"
    })
        ]),
        dcc.Tab(label='Taxi type prediction', children=[
            html.Div([
                html.Div([
                    html.H5(children="Let's use machine learning to predict the color of a given taxi")
                ], style={'marginBottom': 10}
                ),
                dbc.Row([
                    dbc.Col([
                        dcc.Markdown(''' 
                        **Please select the total price of the trip**
                        '''),
                        dcc.Slider(
                            id='price_slider',
                            min=0,
                            max=100,
                            step=1,
                            marks={
                                0: '$0',
                                5: '$5',
                                10: '$10',
                                15: '$15', 
                                20: '$20',
                                30: '$30',
                                50: '$50',
                                100: '$100'
                            },
                            value=10
                        ),
                        dcc.Markdown('''
                        **Please select the pick-up and drop-off zones**
                        '''),
                        dcc.Loading(
                            dcc.Graph(
                                id='choropleth2',
                                figure=dict(
                                    data=[],
                                    layout={},
                                ),
                            )
                        ),  
                    ]),
                    dbc.Col([
                        html.Div([
                            dcc.Store(id='memory', data=[]),
                            dcc.Markdown(id='price_mark',children=
                            '''
                            **Price:**
                            '''),
                            dcc.Markdown(id='plocation',children=
                            '''
                            **Pick-up location:**
                            '''),
                            dcc.Markdown(id='dlocation',children=
                            '''
                            **Drop-off location:**
                            '''),
                            dcc.Markdown("**Your selected taxi is classified as:**"),
                            html.Div([
                                dcc.Markdown(id='ml_result')
                                ] ,id='output_ml'
                            ),
                            html.Div([
                                dcc.Markdown('''
                                **A little bit about our model: **We use a Random Forest Classifier from *sklearn* which 
                                was trained on a balanced dataset. We used a train-test split for the model, with 65% of data
                                being used for training. The model takes only the price, pick-up location and drop-off location as input. Both our training and test accuracy are approximately **87%**.
                                ''')
                            ], style={'marginTop': 60, 'marginRight':35, 'text-align': 'justify'})
                            ], style={'marginTop': 40}
                        )
                    ])
                ])
            ], style={'marginLeft': 30, 'marginTop': 20, 'marginRight': 30}
            )
        ]),
        dcc.Tab(label='About the project', children=[
            html.Div([
            html.H3('Notebook:'),
            dcc.Markdown('''
                The full notebook can be accessed here: 
                [Explainer notebook](https://nbviewer.jupyter.org/github/ewa8/nyctaxis_2020/blob/main/Final_project_notebook_3.ipynb)
            ''')
            ], style={'marginTop': 30, 'marginLeft':30})
        ])
        ], colors={
            "border": "white",
            "primary": "#ea656f",
            "background": "#fff5e6"
    }) 
    ])

@app.callback(
    Output('choropleth', 'figure'), 
    Input('year_slider', 'value'),
    Input('display_selection', 'value'),
    Input('colors', 'value'))
def display_choropleth(day_value, selection, color):
    aggfunc = 'mean'
    zones = 'PULocationID'

    #checking which data to use for displaying the map
    if color == 'yellow':
        filt_data = yellow_data
    elif color == 'green':
        filt_data = green_data
    else:
        filt_data = data

    #if counting the number of pick-ups and drop-offs then use sum and counts    
    if (selection == 'pickup_date' or selection == 'dropoff_date'):
        if selection == 'dropoff_date':
            zones = 'DOLocationID'
        selection = 'count'
        aggfunc = 'sum'


    if selection != 'total':
        filt_data = filt_data[(filt_data['pickup_date'].dt.date < unixToDatetime(day_value).date() + pd.to_timedelta(7, unit='D')) & (filt_data['pickup_date'].dt.date >= unixToDatetime(day_value).date())].groupby(zones).agg(aggfunc)[[selection]]
    #merge the grouped data for pick-up and drop-off locations
    elif selection == 'total':
        pulocation = filt_data[(filt_data['pickup_date'].dt.date < unixToDatetime(day_value).date() + pd.to_timedelta(7, unit='D')) & (filt_data['pickup_date'].dt.date >= unixToDatetime(day_value).date())].groupby('PULocationID').agg('count')[['count']]
        dolocation = filt_data[(filt_data['pickup_date'].dt.date < unixToDatetime(day_value).date() + pd.to_timedelta(7, unit='D')) & (filt_data['pickup_date'].dt.date >= unixToDatetime(day_value).date())].groupby('DOLocationID').agg('count')[['count']]
        merged = pulocation.merge(dolocation, left_index=True, right_index=True, how='outer')
        merged['total'] = merged.sum(axis=1)
        filt_data = merged
    

    fig = px.choropleth_mapbox(
        filt_data, geojson=gj, color=selection,
        locations=filt_data.index, featureidkey="properties.location_id",
        center={"lat": 40.75, "lon": -74}, zoom=10,
        color_continuous_scale = colors_sc[color],
        range_color=[0, max(filt_data[selection])])

    fig.update_traces(hovertemplate=[zones_dict[str(idx)] for idx in filt_data.index])  #+filt_data[selection]) 
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        mapbox_accesstoken=token)

    return fig

@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('year_slider', 'value')])
def update_output(value):
    curr_date = unixToDatetime(value) + pd.to_timedelta(1, unit='D')

    if (curr_date.date() in dates_df['Dates'].dt.date.unique()):
        return '**{0} - {1}**'.format(curr_date.strftime('%Y-%m-%d'), dates_df[dates_df['Dates'].dt.date == curr_date.date()]['Desc'].item())
    else:
        return 'Date selected: **{}**'.format(curr_date.strftime('%Y-%m-%d'))


@app.callback(
    Output('histogram', 'figure'), 
    Input('colors_2', 'value'),
    Input('display_2', 'value'))
def display_hist(color, selection):

    if color == 'yellow':
        filt_data = yellow_data
        hist_color='#FEAF16'
    elif color == 'green':
        filt_data = green_data
        hist_color='#54A24B'
    else:
        filt_data = data
        hist_color='rgb(180,151,231)'


    fig = px.histogram(filt_data, x=selection, log_y=True, hover_data=filt_data.columns, color_discrete_sequence=[hist_color])
    fig.update_layout(margin={"r":0,"t":40,"l":50,"b":0}, title="Distribution of {}".format((title_dict[selection]).lower()), xaxis_title=title_dict[selection])
    return fig

@app.callback(
    Output('line_plot', 'figure'), 
    Input('colors_21', 'value'),
    Input('display_21', 'value'))
def display_line(color, selection):

    if color == 'yellow':
        filt_data = yellow_data
        line_color ='#FECB52'
        dot_color='#edad08'
    elif color == 'green':
        filt_data = green_data
        line_color='#66a61e'
        dot_color='#117733'
    else:
        filt_data = data
        line_color = 'rgb(180,151,231)'
        dot_color = '#cc6677'

    aggfunc = 'sum'
    if selection[0:3] == 'avg':
        aggfunc = 'mean'    
        selection = selection[4:]
    dates_agg = filt_data.groupby('pickup_date').agg(aggfunc)[selection]

    dates_frame = dates_df.merge(dates_agg, left_on='Dates', right_index=True, how='left')
    
    # dates_agg = dates_agg.reset_index()

    # #reset the dates index
    # i = 0
    # j = 7
    # while(True):

    #     day = dates_agg.iloc[i:j, 0].values[0]
    #     dates_agg.iloc[i:j, 0] = day
    #     i += 7
    #     j += 7

    #     if j > len(dates_agg):
    #         day = dates_agg.iloc[i:, 0].values[0]
    #         N = len(dates_agg.iloc[i:, 0])
    #         dates_agg.iloc[i:, 0] = day
    #         break

    # dates_agg = dates_agg.groupby('pickup_date').agg(aggfunc)
    # dates_agg = dates_agg.reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_agg.index, y=dates_agg.values, 
                    mode='lines',
                    line=dict(color=line_color)))

    fig.add_trace(go.Scatter(x=dates_frame['Dates'], y=dates_frame[selection],
                    mode='markers', marker=dict(color=dot_color, size=7),
                    customdata=dates_frame['Desc'],
                    hovertemplate=dates_frame['Desc']))

    fig.update_layout(margin={"r":20,"t":40,"l":0,"b":40}, title="Daily changes in {} ".format((title_dict[selection]).lower()),
        xaxis_title="Dates",
        yaxis_title=title_dict[selection],
        showlegend=False)

    fig.update_yaxes(type="log")
    fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=16
    )
)
    return fig


@app.callback(
    Output('choropleth2', 'figure'),
    Input('choropleth2', 'clickData'),
    Input('memory', 'data'))
def update_figure(clickData, selections):  
    filt_data = ml_data

    fig = px.choropleth_mapbox(filt_data, geojson=gj, color='count',
        locations='zones', featureidkey="properties.location_id",
        center={"lat": 40.75, "lon": -74}, zoom=10, hover_name='zone_name',
        color_continuous_scale = px.colors.sequential.Burgyl,
        range_color=[0, 1],
        opacity=0.6
        )
    #fig.update_traces(hovertemplate=[zones_dict[str(zone)] for zone in filt_data['zones'].astype('int').sort_values().values])

    if len(selections) > 0:
        # highlights contain the geojson information for only 
        # the selected districts
        highlights = get_highlights(selections)

        fig.add_trace(
            px.choropleth_mapbox(filt_data, geojson=highlights, 
                                 color='highlight',
                                 locations='zones', 
                                 featureidkey="properties.location_id", 
                                 hover_name='zone_name',
                                 color_continuous_scale = px.colors.sequential.gray,
                                 range_color=[0, 1],                           
                                 opacity=1).data[0])
        #fig.update_traces(
        #    hovertemplate=[zones_dict[str(zone)] for zone in filt_data['zones'].astype('int').sort_values().values]
        #    )

    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":30},
        mapbox_accesstoken=token,
        coloraxis_showscale=False)

    return fig

@app.callback(
    dash.dependencies.Output('price_mark', 'children'),
    [dash.dependencies.Input('price_slider', 'value')])
def update_price(value):
    return '**Price: ${}**'.format(value)

@app.callback(
    dash.dependencies.Output('plocation', 'children'),
    [dash.dependencies.Input('memory', 'data')])
def update_pulocation(data):
    if len(data) == 1:
        zone_name = zones_dict[str(data[0])]
        return '**Pick-up location: {}**'.format(zone_name)
    else:
        raise PreventUpdate    
    
@app.callback(
    dash.dependencies.Output('dlocation', 'children'),
    dash.dependencies.Input('memory', 'data'))
def update_dolocation(data):
    if len(data) == 2:
        zone_name = zones_dict[str(data[1])]
        return '**Drop-off location: {}**'.format(zone_name)
    else:
        raise PreventUpdate

@app.callback(
    dash.dependencies.Output('memory', 'data'),
    dash.dependencies.Input('choropleth2', 'clickData'),
    State('memory', 'data')
)
def update_store(clickData, data):
    if clickData is not None:            
        location = clickData['points'][0]['location']
        if len(data) < 2:
            data.append(location)
        elif len(data) == 2:
            data = [location]
        return data
    
    else:
        raise PreventUpdate


@app.callback(
    dash.dependencies.Output('ml_result', 'children'),
    dash.dependencies.Input('price_slider', 'value'),
    dash.dependencies.Input('memory', 'data'))
def predict_color(price, data):
    if len(data) == 2:
        result = loaded_model.predict(np.array([data[0], data[1], price]).reshape(1,-1))
        result = colors_res[str(result[0])]
        return '**{}**'.format(result)
    else:
        raise PreventUpdate

@app.callback(
    dash.dependencies.Output('output_ml', 'style'),
    dash.dependencies.Input('ml_result', 'children'))
def predict_color(result):
    result = result[2:-2]
    if result == 'Yellow':
        result = 'gold'
    return {'color':result}

# Stop with ctrl + alt + M
if __name__ == '__main__':
    app.run_server(debug=False)