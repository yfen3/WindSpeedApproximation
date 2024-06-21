import plotly.express as px
import plotly.graph_objects as go
from dash import dcc

# Display a coutour graph
# Modified from source: https://plotly.com/python/contour-plots/
def get_graph(title, latitudes, longitudes, bounds):

    fig = go.Figure(
        data = go.Contour(
            x = longitudes,
            y = latitudes,
            z = bounds,
            contours = dict(
            coloring ='heatmap',
            showlabels = True, # show labels on contours
            labelfont = 
                dict( # label font properties
                    size = 8,
                    color = 'white',
                )
            ),
        ),
    )

    fig.update_layout(
        title=title,
        width=500,
        height=500,
        )

    graph = dcc.Graph(
        figure=fig
    )

    return graph