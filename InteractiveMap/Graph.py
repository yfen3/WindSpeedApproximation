import pandas as pd 
import plotly.express as px
from dash import dcc


# TODO implement the graph
# This is a placeholder for the graph
def get_graph(title):
    df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig = px.bar(
        df, 
        x="Fruit", 
        y="Amount", 
        color="City", 
        barmode="group", 
        title=title,
        width=500
        )
    graph = dcc.Graph(
        id='example-graph',
        figure=fig
    )

    return graph