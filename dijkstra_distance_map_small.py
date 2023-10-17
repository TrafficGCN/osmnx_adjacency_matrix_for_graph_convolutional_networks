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
        sensor_list.append(Point(row["LONGITUDE"], row["LATITUDE"]))
    sensor_centroid = geopandas.GeoSeries(sensor_list).centroid.values[0]

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


# Interface to OSMNX
def generate_paths(df, graph):

    # Create the lists for storing the paths
    long_paths = []
    lat_paths = []
    target_points = []

    # Setup the target points
    for lo, la in zip(df["LONGITUDE"], df["LATITUDE"]):
        target_points.append((la, lo))

    # origin = (df['LATITUDE'][15], df['LONGITUDE'][15])

    for index, row in df.iterrows():

        # Setup the origin point
        origin = (row['LATITUDE'], row['LONGITUDE'])

        # Query the paths
        i = 0
        for target_point in target_points:

            # Process the optimal path
            print("Processing *************************************** " + str(i))

            try:
                # Get the nearest node in the OSMNX graph for the origin point
                origin_node = ox.get_nearest_node(graph, origin)

                # Get the nearest node in the OSMNX graph for the target point
                target_node = ox.get_nearest_node(graph, target_point)

                # Get the optimal path via dijkstra
                route = nx.shortest_path(
                    graph, origin_node, target_node, weight='length', method='dijkstra')

                # Create the arrays for storing the paths
                lat = []
                long = []

                for i in route:
                    point = graph.nodes[i]
                    long.append(point['x'])
                    lat.append(point['y'])

                # Append the paths
                long_paths.append(long)
                lat_paths.append(lat)
            except nx.NetworkXNoPath:
                # Set an empty path
                long_paths.append([])
                lat_paths.append([])

            i += 1

    # long_paths, lat_paths = adjust_path_coordinates(long_paths, lat_paths)

    # Return the paths
    return long_paths, lat_paths


# Plot the results using mapbox and plotly
def plot_map(df, long, lat):

    # Calculate the centroid of the sensor list
    sensor_centroid = sensor_list_centroid(df)

    # Adjust the overlapping segments
    long, lat = adjust_overlapping_segments(
        long_paths, lat_paths, adjustment_amount=0.00006)

    # Create a plotly map and add the origin point to the map
    print("Setting up figure...")
    fig = go.Figure(go.Scattermapbox(
    )
    )

    # Plot the optimal paths to the map
    used_colors = set()
    for i in range(len(lat)):
        while True:
            r = random.randrange(256)
            g = random.randrange(256)
            b = random.randrange(256)
            color = f"#{r:02x}{g:02x}{b:02x}"
            if color not in used_colors:
                used_colors.add(color)
                break

        fig.add_trace(go.Scattermapbox(
            name="Path",
            mode="lines",
            lon=long[i],
            lat=lat[i],
            marker={'size': 10},
            showlegend=False,
            line=dict(width=1, color=color))
        )

    # Plot the sensors to the map
    for index, row in df.iterrows():
        fig.add_trace(go.Scattermapbox(
            name="Destination",
            mode="markers",
            showlegend=False,
            lon=df["LONGITUDE"],
            lat=df["LATITUDE"],
            marker={"size": 25, "color": 'rgba(3,3,3,0.5)', },
            opacity=0.5))

    # Style the map layout
    fig.update_layout(
        mapbox_style="light",
        mapbox_accesstoken="##################",
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
        'zoom': 13.25}
    )

    # Save map in output folder
    print("Saving image to output folder...")
    fig.write_image(OS_PATH + '/output/munich/munich_dijkstra_map_bike_small.jpg', scale=3)

    # Show the map in the web browser
    print("Generating map in browser...")
    # fig.show()


def adjust_overlapping_segments(long_paths, lat_paths, adjustment_amount):
    """
    This function takes a list of longitudes and latitudes for different paths and
    adjusts the geocoordinates slightly wherever they overlap so they are all visible
    side by side without any overlapping.

    Parameters:
    -----------
    long_paths : list of lists
        A list of lists containing longitudes for different paths.
    lat_paths : list of lists
        A list of lists containing latitudes for different paths.
    adjustment_amount : float
        The amount by which to adjust the overlapping geocoordinates.

    Returns:
    --------
    adjusted_long_paths : list of lists
        A list of lists containing adjusted longitudes for different paths.
    adjusted_lat_paths : list of lists
        A list of lists containing adjusted latitudes for different paths.
    """
    adjusted_long_paths = []
    adjusted_lat_paths = []

    for i in range(len(long_paths)):
        # check if this path overlaps with any other path
        for j in range(i+1, len(long_paths)):
            long_i = long_paths[i]
            lat_i = lat_paths[i]
            long_j = long_paths[j]
            lat_j = lat_paths[j]

            # check if the paths have any common points
            common_points = set(zip(long_i, lat_i)).intersection(
                set(zip(long_j, lat_j)))

            if len(common_points) > 0:
                # shift the overlapping points by a small amount to avoid overlap
                for point in common_points:
                    index_i = list(zip(long_i, lat_i)).index(point)
                    index_j = list(zip(long_j, lat_j)).index(point)
                    long_i[index_i] -= adjustment_amount
                    lat_i[index_i] -= adjustment_amount
                    long_j[index_j] += adjustment_amount
                    lat_j[index_j] += adjustment_amount

        # append the adjusted paths to the list
        adjusted_long_paths.append(long_i)
        adjusted_lat_paths.append(lat_i)

    return adjusted_long_paths, adjusted_lat_paths


# MAIN

# Data import path
# Data import path
OS_PATH = os.path.dirname(os.path.realpath('__file__'))
SENSORS_CSV = os.path.join(OS_PATH, 'data/munich/munich_bicycle_sensors.csv')

# Data Import
df = pd.read_csv(SENSORS_CSV)

# Keep only relevant columns
df = df.loc[:, ("DETEKTOR_ID", "LATITUDE", "LONGITUDE")]

# Create point geometries
geometry = geopandas.points_from_xy(df.LONGITUDE, df.LATITUDE)
geo_df = geopandas.GeoDataFrame(
    df[['DETEKTOR_ID', 'LATITUDE', 'LONGITUDE']], geometry=geometry)

# Create the networkx graph
# 'drive', 'bike', 'walk'
graph = create_graph("Munich, Bavaria, Germany", "bike")

long_paths, lat_paths = generate_paths(df, graph)

plot_map(df, long_paths, lat_paths)
