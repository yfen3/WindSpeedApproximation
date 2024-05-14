import dash_leaflet as dl

#create the info panel where our wind speed will be displayed 
# Retrun a basic info panel
def get_tooltip(position, content='Placeholder tooltip text'):
    tooltip = dl.Marker(
        id='tooltip',
        position=position, 
        )
    
    return tooltip