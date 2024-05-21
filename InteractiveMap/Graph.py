import plotly.express as px
from dash import dcc


# TODO implement the graph
# This is a placeholder for the graph
def get_graph(title, df, x, y):

    fig = px.line(
        df, 
        x=x, 
        y=y, 
        title=title,
        markers=True,
        width=500
        )
    graph = dcc.Graph(
        figure=fig
    )

    return graph