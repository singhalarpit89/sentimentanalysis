import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dash_table
from dash.exceptions import PreventUpdate
from io import StringIO
import pycountry

# Load the data
df = pd.read_csv('bayport_social_sentiment_synthetic.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Convert country names to ISO3 format
def get_iso3(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except LookupError:
        return None

df['country_iso3'] = df['country'].apply(get_iso3)

# Color palette
colors = {
    'primary': '#003f5c',
    'secondary': '#58508d',
    'accent1': '#bc5090',
    'accent2': '#ff6361',
    'accent3': '#ffa600',
    'background': '#f8f9fa',
    'text': '#212529',
    'positive': '#168AAD',
    'neutral': '#8A8A8A',
    'negative': '#D00000'
}

external_stylesheets = [dbc.themes.FLATLY, "https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "BayPort Financial - Social Sentiment Analysis"

# KPIs
total_negative = df[df['sentiment_category'] == 'Negative'].shape[0]
average_sentiment = df['sentiment_polarity'].mean()
launch_date = pd.to_datetime('2025-07-01')
def classify_period(x):
    if x.date() < launch_date.date():
        return 'Pre-Launch'
    else:
        return 'Post-Launch'
df['Launch_Period'] = df['datetime'].apply(classify_period)
periods = ['Pre-Launch', 'Post-Launch']
negative_percentages = {
    period: df[df['Launch_Period'] == period]['sentiment_category'].value_counts(normalize=True).get('Negative', 0) * 100
    for period in periods
}

header = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="https://via.placeholder.com/50x50?text=BF", height="50px")),
                        dbc.Col(dbc.NavbarBrand("BayPort Financial", className="ml-2", style={"fontSize": "24px", "fontFamily": "Montserrat, sans-serif"})),
                    ],
                    align="center",
                ),
                href="/",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Dashboard", href="#")),
                        dbc.NavItem(
                            dbc.Button("Export Data", id="export-button", color="success", className="ml-2")
                        ),
                    ],
                    className="ml-auto",
                    navbar=True,
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ]
    ),
    color=colors['primary'],
    dark=True,
    className="mb-4",
)

app.layout = html.Div([
    header,
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-kpis",
                    type="default",
                    children=[
                        dbc.Card([
                            dbc.CardHeader("Total Negative Feedback", className="text-center"),
                            dbc.CardBody([
                                html.H3(f"{total_negative}", className="card-title text-center"),
                                html.P("Messages requiring attention", className="card-text text-center")
                            ])
                        ], className="mb-4", style={"borderTop": f"3px solid {colors['negative']}"})
                    ]
                )
            ], width=4),
            dbc.Col([
                dcc.Loading(
                    id="loading-avg-sentiment",
                    type="default",
                    children=[
                        dbc.Card([
                            dbc.CardHeader("Average Sentiment Score", className="text-center"),
                            dbc.CardBody([
                                html.H3(f"{average_sentiment:.2f}", className="card-title text-center"),
                                html.P("Overall customer sentiment", className="card-text text-center")
                            ])
                        ], className="mb-4", style={"borderTop": f"3px solid {colors['positive']}"})
                    ]
                )
            ], width=4),
            dbc.Col([
                dcc.Loading(
                    id="loading-launch",
                    type="default",
                    children=[
                        dbc.Card([
                            dbc.CardHeader("Launch Period Negative %", className="text-center"),
                            dbc.CardBody([
                                html.H3(f"{negative_percentages['Post-Launch']:.1f}%", className="card-title text-center"),
                                html.P("Impact of recent product launch", className="card-text text-center")
                            ])
                        ], className="mb-4", style={"borderTop": f"3px solid {colors['accent3']}"})
                    ]
                )
            ], width=4),
        ]),

        dbc.Card([
            dbc.CardHeader("Filters", style={"backgroundColor": colors['primary'], "color": "white"}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id='date-range',
                            start_date=df['datetime'].min().date(),
                            end_date=df['datetime'].max().date(),
                            display_format='YYYY-MM-DD',
                            style={"width": "100%"}
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label("Platform:"),
                        dcc.Dropdown(
                            id='platform-filter',
                            options=[{'label': platform, 'value': platform} for platform in df['platform'].unique()],
                            value=None,
                            placeholder="All Platforms",
                            style={"color": "#000"}
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label("Country:"),
                        dcc.Dropdown(
                            id='country-filter',
                            options=[{'label': country, 'value': country} for country in df['country'].unique()],
                            value=None,
                            placeholder="All Countries",
                            style={"color": "#000"}
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label("Topic:"),
                        dcc.Dropdown(
                            id='topic-filter',
                            options=[{'label': topic, 'value': topic} for topic in df['topic'].unique()],
                            value=None,
                            placeholder="All Topics",
                            style={"color": "#000"}
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label("Launch Period:"),
                        dcc.Dropdown(
                            id='launch-period-filter',
                            options=[
                                {'label': 'Pre-Launch', 'value': 'Pre-Launch'},
                                {'label': 'Post-Launch', 'value': 'Post-Launch'}
                            ],
                            value=None,
                            placeholder="All Periods",
                            style={"color": "#000"}
                        ),
                    ], width=3),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(
                            dbc.Button("Apply Filters", id="apply-filters", color="primary", className="mt-3"),
                            className="text-center"
                        )
                    ], width=12)
                ])
            ])
        ], className="mb-4"),

        dbc.Tabs([
            dbc.Tab(label="Sentiment Analysis", tab_id="overview", children=[
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-trend",
                            type="circle",
                            children=[dcc.Graph(id='sentiment-trend')]
                        )
                    ], width=6),
                    dbc.Col([
                        dcc.Loading(
                            id="loading-platform",
                            type="circle",
                            children=[dcc.Graph(id='platform-comparison')]
                        )
                    ], width=6),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-dist",
                            type="circle",
                            children=[dcc.Graph(id='sentiment-distribution')]
                        )
                    ], width=6),
                    dbc.Col([
                        dcc.Loading(
                            id="loading-keyword",
                            type="circle",
                            children=[dcc.Graph(id='keyword-analysis')]
                        )
                    ], width=6),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-geo",
                            type="circle",
                            children=[dcc.Graph(id='geo-sentiment')]
                        )
                    ], width=12),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-topic",
                            type="circle",
                            children=[dcc.Graph(id='topic-sentiment')]
                        )
                    ], width=12),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-language",
                            type="circle",
                            children=[dcc.Graph(id='language-distribution')]
                        )
                    ], width=12),
                ], className="mb-4"),
            ]),
            dbc.Tab(label="Message Table", tab_id="table", children=[
                dbc.Row([
                    dbc.Col([
                        html.H3("Social Feedback Details", className="text-center mb-4"),
                        dcc.Loading(
                            id="loading-table",
                            type="circle",
                            children=[
                                dash_table.DataTable(
                                    id='message-table',
                                    columns=[
                                        {"name": "Date", "id": "datetime"},
                                        {"name": "Platform", "id": "platform"},
                                        {"name": "Country", "id": "country"},
                                        {"name": "Language", "id": "language"},
                                        {"name": "Topic", "id": "topic"},
                                        {"name": "Sentiment", "id": "sentiment_category"},
                                        {"name": "Polarity", "id": "sentiment_polarity"},
                                        {"name": "Intensity", "id": "sentiment_intensity"},
                                        {"name": "Emotion", "id": "emotion"},
                                        {"name": "Likes", "id": "engagement_likes"},
                                        {"name": "Shares", "id": "engagement_shares"},
                                        {"name": "Comments", "id": "engagement_comments"},
                                        {"name": "Influencer Level", "id": "influencer_level"},
                                        {"name": "Keywords", "id": "keywords"},
                                        {"name": "Risk Flag", "id": "risk_flag"}
                                    ],
                                    page_size=10,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"textAlign": "left", "padding": "10px"},
                                    style_header={
                                        "backgroundColor": colors['primary'],
                                        "color": "white",
                                        "fontWeight": "bold"
                                    },
                                    style_data_conditional=[
                                        {
                                            'if': {'filter_query': '{sentiment_category} = "Negative"'},
                                            'backgroundColor': '#ffebee',
                                            'color': '#c62828'
                                        },
                                        {
                                            'if': {'filter_query': '{sentiment_category} = "Positive"'},
                                            'backgroundColor': '#e8f5e9',
                                            'color': '#2e7d32'
                                        }
                                    ],
                                    filter_action="native",
                                    sort_action="native",
                                    sort_mode="multi",
                                    row_selectable="multi",
                                    selected_rows=[],
                                    export_format="csv"
                                )
                            ]
                        )
                    ], width=12)
                ], className="mb-4")
            ]),
        ], id="tabs", active_tab="overview"),
    ], fluid=True),

    html.Footer(
        dbc.Container([
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("BayPort Financial Social Sentiment Dashboard © 2025", className="text-center text-muted")
                ], width=12)
            ])
        ]),
        style={"marginTop": "40px", "paddingBottom": "20px"}
    ),
    dcc.Store(id='filtered-data'),
    dcc.Download(id="download-dataframe-csv"),
])

@app.callback(
    Output('filtered-data', 'data'),
    [Input('apply-filters', 'n_clicks')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('platform-filter', 'value'),
     State('country-filter', 'value'),
     State('topic-filter', 'value'),
     State('launch-period-filter', 'value')]
)
def filter_data(n_clicks, start_date, end_date, platform, country, topic, launch_period):
    if n_clicks is None:
        return df.to_json(date_format='iso', orient='split')
    filtered_df = df.copy()
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['datetime'].dt.date >= pd.to_datetime(start_date).date()) & (filtered_df['datetime'].dt.date <= pd.to_datetime(end_date).date())]
    if platform:
        filtered_df = filtered_df[filtered_df['platform'] == platform]
    if country:
        filtered_df = filtered_df[filtered_df['country'] == country]
    if topic:
        filtered_df = filtered_df[filtered_df['topic'] == topic]
    if launch_period:
        filtered_df = filtered_df[filtered_df['Launch_Period'] == launch_period]
    return filtered_df.to_json(date_format='iso', orient='split')

@app.callback(
    [
        Output('sentiment-trend', 'figure'),
        Output('platform-comparison', 'figure'),
        Output('sentiment-distribution', 'figure'),
        Output('keyword-analysis', 'figure'),
        Output('geo-sentiment', 'figure'),
        Output('message-table', 'data'),
        Output('language-distribution', 'figure'),
        Output('topic-sentiment', 'figure')
    ],
    [Input('filtered-data', 'data')]
)
def update_charts(filtered_data):
    if filtered_data is None:
        raise PreventUpdate
    filtered_df = pd.read_json(StringIO(filtered_data), orient='split')
    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(
            title="No Data Available with Current Filters",
            annotations=[{"text": "Try adjusting your filters", "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False}]
        )
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, [], empty_fig, empty_fig

    # 1. Sentiment Trend
    sentiment_trend = filtered_df.copy()
    sentiment_trend['date'] = sentiment_trend['datetime'].dt.date
    daily_trend = sentiment_trend.groupby('date')['sentiment_polarity'].mean().reset_index()
    daily_trend['moving_avg'] = daily_trend['sentiment_polarity'].rolling(window=14, min_periods=1).mean()
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=daily_trend['date'],
        y=daily_trend['moving_avg'],
        mode='lines+markers',
        name='14-Day Moving Avg Sentiment',
        line=dict(width=3, color=colors['positive']),
        fill='tozeroy',
        fillcolor=f"rgba({int(colors['positive'][1:3], 16)}, {int(colors['positive'][3:5], 16)}, {int(colors['positive'][5:7], 16)}, 0.2)"
    ))
    trend_fig.update_layout(
        title="Smoothed Sentiment Polarity Trend Over Time",
        xaxis_title="Date",
        yaxis_title="Moving Avg Sentiment Polarity",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Montserrat, sans-serif"),
        hovermode="x unified",
        margin=dict(t=60, b=40, l=40, r=30)
    )

    # 2. Platform Comparison
    platform_comparison = filtered_df.groupby('platform')['sentiment_polarity'].mean().reset_index()
    platform_fig = go.Figure()
    for idx, row in platform_comparison.iterrows():
        color = colors['positive'] if row['sentiment_polarity'] > 0 else colors['negative']
        platform_fig.add_trace(go.Bar(
            y=[row['platform']],
            x=[row['sentiment_polarity']],
            orientation='h',
            marker=dict(
                color=color,
                line=dict(color=color, width=1)
            ),
            name=row['platform']
        ))
    platform_fig.update_layout(
        title="Average Sentiment Polarity by Platform",
        xaxis_title="Average Sentiment Polarity",
        yaxis_title="Platform",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Montserrat, sans-serif"),
        margin=dict(t=60, b=40, l=100, r=30),
        showlegend=False
    )

    # 3. Sentiment Distribution
    sentiment_counts = filtered_df['sentiment_category'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment_category', 'count']
    sentiment_colors = {
        'Positive': colors['positive'],
        'Neutral': colors['neutral'],
        'Negative': colors['negative']
    }
    color_list = [sentiment_colors.get(cat, colors['neutral']) for cat in sentiment_counts['sentiment_category']]
    sentiment_fig = go.Figure()
    sentiment_fig.add_trace(go.Pie(
        labels=sentiment_counts['sentiment_category'],
        values=sentiment_counts['count'],
        hole=0.6,
        marker=dict(colors=color_list),
        textinfo='percent+label',
        insidetextorientation='radial'
    ))
    sentiment_fig.update_layout(
        title="Sentiment Distribution",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Montserrat, sans-serif"),
        margin=dict(t=60, b=40, l=40, r=30),
        legend=dict(orientation="h", y=-0.1)
    )
    sentiment_fig.add_annotation(
        text=f"Total<br>{sentiment_counts['count'].sum()}",
        x=0.5, y=0.5,
        font_size=16,
        showarrow=False
    )

    # 4. Keyword Analysis
    keyword_counts = filtered_df['keywords'].value_counts().reset_index()
    keyword_counts.columns = ['Keyword', 'count']
    keyword_fig = px.treemap(
        keyword_counts,
        path=['Keyword'],
        values='count',
        color='count',
        color_continuous_scale=[colors['primary'], colors['accent1'], colors['accent3']],
        title="Top Keywords in Social Feedback"
    )
    keyword_fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Montserrat, sans-serif"),
        margin=dict(t=60, b=40, l=40, r=30)
    )

    # 5. Geo Sentiment
    geo_sentiment = filtered_df.groupby('country_iso3')['sentiment_polarity'].mean().reset_index()
    geo_fig = px.choropleth(
        geo_sentiment,
        locations="country_iso3",
        locationmode="ISO-3",
        color="sentiment_polarity",
        color_continuous_scale=[colors['negative'], colors['neutral'], colors['positive']],
        title="Average Sentiment Polarity by Country"
    )
    geo_fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Montserrat, sans-serif"),
        margin=dict(t=60, b=40, l=40, r=30)
    )

    # 6. Language Distribution
    language_counts = filtered_df['language'].value_counts().reset_index()
    language_counts.columns = ['language', 'count']
    language_fig = px.pie(
        language_counts,
        names='language',
        values='count',
        title="Feedback Distribution by Language"
    )
    language_fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Montserrat, sans-serif"),
        margin=dict(t=60, b=40, l=40, r=30)
    )

    # 7. Topic Sentiment
    topic_sentiment = filtered_df.groupby('topic')['sentiment_polarity'].mean().reset_index()
    topic_fig = px.bar(
        topic_sentiment,
        x='topic',
        y='sentiment_polarity',
        color='sentiment_polarity',
        color_continuous_scale=[colors['negative'], colors['neutral'], colors['positive']],
        title="Average Sentiment Polarity by Topic"
    )
    topic_fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(family="Montserrat, sans-serif"),
        margin=dict(t=60, b=40, l=40, r=30)
    )

    # 8. Table Data
    table_data = filtered_df.to_dict('records')
    return trend_fig, platform_fig, sentiment_fig, keyword_fig, geo_fig, table_data, language_fig, topic_fig

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-button", "n_clicks"),
    State("filtered-data", "data"),
    prevent_initial_call=True,
)
def export_data(n_clicks, filtered_data):
    if filtered_data is None:
        return None
    filtered_df = pd.read_json(filtered_data, orient='split')
    return dcc.send_data_frame(filtered_df.to_csv, "social_sentiment_export.csv")

if __name__ == '__main__':
    app.run(debug=True)