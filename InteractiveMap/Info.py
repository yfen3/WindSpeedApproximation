from dash import html

#create the info panel where our wind speed will be displayed 
# Retrun a basic info panel
def get_info_panel(placeholder, style):
    info = html.Div(id="info",
                className="info",
                style=style,
                children=placeholder
                )
    
    return info