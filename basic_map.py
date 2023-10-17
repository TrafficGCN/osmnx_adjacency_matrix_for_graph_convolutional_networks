import os
import random
import networkx as nx
import plotly.graph_objects as go
import osmnx as ox
import pandas as pd
import geopandas
from shapely.geometry import Point


# Using the cache accelerates processing for a large map
ox.config(log_console=True, use_cache=True, timeout=1000)

# Create a graph based on the mode of travel and place


def graph_centroid(place):
    # get the boundary of the place as a polygon
    place_polygon = ox.geocode_to_gdf(place).unary_union

    # calculate the centroid of the polygon
    centroid = place_polygon.centroid

    return centroid


def sensor_list_centroid(df):
    # Calculate the centroid of the sensor list
    sensor_list = []
    for index, row in df.iterrows():
        sensor_list.append(Point(row["long"], row["lat"]))
    sensor_centroid = geopandas.GeoSeries(sensor_list).centroid.values[0]

    print(sensor_centroid)

    return sensor_centroid


def create_graph(place, mode):

    # request the graph from the inputs
    graph = ox.graph_from_place(place, network_type=mode)

    # get the largest strongly connected component
    scc_generator = nx.strongly_connected_components(graph)
    largest_scc = max(scc_generator, key=len)

    # create a new graph containing only the largest strongly connected component
    graph = graph.subgraph(largest_scc)

    return graph

# Plot the results using mapbox and plotly
def plot_map(df):

    # Calculate the centroid of the sensor list
    sensor_centroid = sensor_list_centroid(df)

    # Create a plotly map and add the origin point to the map
    print("Setting up figure...")
    fig = go.Figure(go.Scattermapbox(
    )
    )

    # Plot the sensors to the map
    for index, row in df.iterrows():
        fig.add_trace(go.Scattermapbox(
            name="Destination",
            mode="markers",
            showlegend=False,
            lon=df["long"],
            lat=df["lat"],
            marker={"size": 7, "color": 'rgba(3,3,3,0.5)', },
            opacity=0.5))

    # Style the map layout
    fig.update_layout(
        mapbox_style="light",
        mapbox_accesstoken="",
        legend=dict(yanchor="top", y=1, xanchor="left", x=0.83),  # x 0.9
        # title="<span style='font-size: 32px;'><b>The Shortest Paths Dijkstra Map</b></span>",
        font_family="Times New Roman",
        font_color="#333333",
        title_font_size=32,
        font_size=18,
        width=2000,  # 2000
        height=2000,
    )

    # Add the center to the map layout
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      title=dict(yanchor="top", y=.97,
                                 xanchor="left", x=0.03),  # x 0.75
                      mapbox={
        'center': {'lat': sensor_centroid.y,
                   'lon': sensor_centroid.x},
        'zoom': 11.5}
    )

    # Save map in output folder
    print("Saving image to output folder...")
    #fig.write_image(OS_PATH + '/output/utd-los-angeles/basic_map_2.jpg', scale=3)

    # Show the map in the web browser
    print("Generating map in browser...")
    fig.show()


# MAIN

# Data import path
# Data import path
OS_PATH = os.path.dirname(os.path.realpath('__file__'))
SENSORS_CSV = os.path.join(OS_PATH, 'data/utd-los-angeles/los_angeles_sensors.csv')

# Data Import
df = pd.read_csv(SENSORS_CSV)

# Keep only relevant columns
df = df.loc[:, ("detid", "lat", "long")]

# Create point geometries
geometry = geopandas.points_from_xy(df.long, df.lat)
geo_df = geopandas.GeoDataFrame(
    df[['detid', 'lat', 'long']], geometry=geometry)

# Create the networkx graph
# 'drive', 'bike', 'walk'
graph = create_graph("Los Angeles, California, USA", "drive")  # Using San Francisco for demonstration


plot_map(df) 