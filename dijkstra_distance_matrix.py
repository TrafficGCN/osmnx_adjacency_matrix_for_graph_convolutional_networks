'''
The idea was sparked from Wei-Meng Lee on towardsdatascience
https://towardsdatascience.com/visualization-in-python-finding-routes-between-points-2d97d4881996

For graph neural networks fastest Dijkstra routes can be found using OpenStreetMap API. 

Multi processing on a CPU or using a GPU is advised as the the process is very resource intensive.
place = 'Munich, Bavaria, Germany'

Using the six sensors from the RAD dataset as an example to connect the six sensors over the road network
'''

import os
import numpy as np
from itertools import islice
import pandas as pd
import osmnx as ox
import networkx as nx

# find shortest route based on the mode of travel and place
def create_graph(place, mode):

    # request the graph from the inputs
    graph = ox.graph_from_place(place, network_type=mode)

    # get the largest strongly connected component
    scc_generator = nx.strongly_connected_components(graph)
    largest_scc = max(scc_generator, key=len)

    # create a new graph containing only the largest strongly connected component
    graph = graph.subgraph(largest_scc)

    return graph

##### Interface to OSMNX
def generate_adjacency_matrix(df):

    # Create the adjacency matrix
    matrix = [["DETEKTOR_ID_X, DETEKTOR_ID_Y, DISTANCE"]]

    # nested for loop to find the distances between all the sensors for every sensor
    i=0;
    for index, detector in islice(df.iterrows(), 0, len(list(df.DETEKTOR_ID))):
        j=0;
        for l, each_detector in df.iterrows():

            # coordinates from the current sensor
            start_latlng = (float(detector["LATITUDE"]), float(detector["LONGITUDE"]))
            # coordinates belonging to the destination sensor
            end_latlng = (float(each_detector["LATITUDE"]), float(each_detector["LONGITUDE"]))

            # Try catch because sometimes a node isn't on the graph
            try:
                # find the nearest node to the current sensor
                orig_node = ox.get_nearest_node(graph, start_latlng)
                # find the nearest node to the destination sensor
                dest_node = ox.get_nearest_node(graph, end_latlng)

                #find the shortest path method dijkstra or bellman-ford
                shortest_route_distance = nx.shortest_path_length(graph, orig_node,dest_node, weight="length", method="dijkstra")
            except nx.NetworkXNoPath:
                shortest_route_distance = 0
    
            matrix.append([str(detector["DETEKTOR_ID"]) + "," + str(each_detector["DETEKTOR_ID"]) + "," + str(float(shortest_route_distance))])

            #print(matrix)
            
            j=+1;
        
        i=+1;

    matrix = np.array(matrix)
    matrix = np.asarray(matrix)

    # Save the dijkstra rad sensors distance matrix
    np.savetxt(OS_PATH + "output/munich/bicycle_adjacency_matrix.csv", matrix, delimiter=",", fmt='%s')
    
def transpose_adjacency_matrix(adjacency_matrix_file):
    # Read the adjacency matrix CSV file into a pandas DataFrame
    df = pd.read_csv(adjacency_matrix_file)
    
    df['DISTANCE'] = df['DISTANCE'].astype(float)

    # Transpose the matrix using pivot_table()
    df_transposed = df.pivot_table(index='detid_Y', columns='detid_X', values='DISTANCE')

    # Set the diagonal elements to zero
    np.fill_diagonal(df_transposed.values, 0)

    # Save the transposed matrix as a CSV file
    df_transposed.to_csv(os.path.splitext(adjacency_matrix_file)[0] + '_transposed.csv', float_format='%.2f')


def normalize_matrix(transposed_matrix_file):
    # Read the transposed adjacency matrix CSV file into a pandas DataFrame

    df = pd.read_csv(transposed_matrix_file, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')  # convert strings to numbers
    

    # Find the minimum and maximum values in the matrix
    min_value = df.min()
    max_value = df.max()

    # Normalize the matrix values between 0 and 1
    df_normalized = (df - min_value) / (max_value - min_value)

    # Save the transposed matrix as a CSV file
    df_normalized.to_csv(os.path.splitext(transposed_matrix_file)[0] + '_normalized.csv')


# Data import path
OS_PATH = os.path.dirname(os.path.realpath('__file__'))
SENSORS_CSV   = OS_PATH + '/data/munich/munich_bicycle_sensors.csv'

# Data Import Path
df = pd.read_csv(SENSORS_CSV)

# Keep only relevant columns
df = df.loc[:, ("DETEKTOR_ID","LATITUDE", "LONGITUDE")]

# Remove missing geocoordinates
df.dropna(subset=['LATITUDE'], how='all', inplace=True)
df.dropna(subset=['LONGITUDE'], how='all', inplace=True)

# Remove missing sensor ids
df.dropna(subset=['DETEKTOR_ID'], how='all', inplace=True)

# Create the networkx graph ONLY CREATE THIS ONCE TO REDUCE OSM REQUESTS!!!
# 'drive', 'bike', 'walk'
graph = create_graph("Munich, Bavaria, Germany", "bike")

generate_adjacency_matrix(df)

# Transpose the combined adjacency matrix and save it as a CSV file
adjacency_matrix_file = os.path.join(OS_PATH, '/output/munich/bicycle_adjacency_matrix.csv')
transpose_adjacency_matrix(adjacency_matrix_file)

transposed_matrix_file = os.path.join(OS_PATH, '/output/munich/bicycle_adjacency_matrix_transposed.csv')
normalize_matrix(transposed_matrix_file)
