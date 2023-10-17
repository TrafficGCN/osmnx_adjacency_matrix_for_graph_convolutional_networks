'''
Run this code on Google Cloud Services or Amazon Web Services for faster processing. A process for each row in the Dataframe is created.

Multi processing on a CPU or using a GPU is advised as the the process is very resource intensive.

For graph neural networks optimal Dijkstra routes can be found using OpenStreetMap API.
e.g. https://towardsdatascience.com/visualization-in-python-finding-routes-between-points-2d97d4881996

Using the six sensors from the RAD dataset as an example to connect the six sensors over the road network
'''
import multiprocessing as mp
import os
import numpy as np
from itertools import islice
import pandas as pd
import osmnx as ox
import networkx as nx

# Using the cache accelerates processing for a large map
ox.config(log_console=True, use_cache=True, timeout=1000)

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
def generate_adjacency_matrix(df, df_row, row_index, graph):

    # Data import path
    OS_PATH = os.path.dirname(os.path.realpath('__file__'))

    # Create the adjacency matrix
    matrix = [["detid_X,detid_Y,DISTANCE"]]

    # nested for loop to find the distances between all the sensors for every sensor
    i=0;
    for index, detector in islice(df_row.iterrows(), 0, len(list(df_row.detid))):
        j=0;
        for l, each_detector in df.iterrows():

            # coordinates from the current sensor
            start_latlng = (float(detector["lat"]), float(detector["long"]))
            # coordinates belonging to the destination sensor
            end_latlng = (float(each_detector["lat"]), float(each_detector["long"]))

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
    

            matrix.append([str(detector["detid"]) + "," + str(each_detector["detid"]) + "," + str(float(shortest_route_distance))])

            print(matrix)
            
            j=+1;
        
        i=+1;

    matrix = np.array(matrix)
    matrix = np.asarray(matrix)

    # Save the dijkstra rad sensors distance matrix
    np.savetxt(OS_PATH + "/output/munich_adjacency_matrix_" + str(row_index) + ".csv", matrix, delimiter=",", fmt='%s')

def combine_csv_files(df):

    # Data import path
    OS_PATH = os.path.dirname(os.path.realpath('__file__'))

    # Create an empty list to store dataframes
    dfs = []

    # Loop through all the CSV files created by generate_adjacency_matrix
    for i in range(1, len(df) + 1):
        file_path = OS_PATH + "/output/munich_adjacency_matrix_" + str(i) + ".csv"
        if os.path.exists(file_path):
            # Read the CSV file into a dataframe and append it to the list
            df = pd.read_csv(file_path)
            dfs.append(df)

    # Concatenate all the dataframes in the list into a single dataframe
    result = pd.concat(dfs)

    # Save the concatenated dataframe as a CSV file
    result.to_csv(OS_PATH + "/output/munich_adjacency_matrix.csv", index=False)


def divide_dataframe(df, num_processes):
    """
    Divide a DataFrame into roughly equal chunks for parallel processing.

    Args:
        df (pd.DataFrame): DataFrame to divide.
        num_processes (int): Number of processes to divide the DataFrame into.

    Returns:
        List of DataFrames, each representing a chunk of the original DataFrame.
    """
    chunk_size = int(np.ceil(len(df) / num_processes))
    return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

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

if __name__ == '__main__':
    # Data import path
    OS_PATH = os.path.dirname(os.path.realpath('__file__'))
    SENSORS_CSV = os.path.join(OS_PATH, '/data/munich/munich_sensors.csv')

    # Data Import Path
    df = pd.read_csv(SENSORS_CSV)

    # Keep only relevant columns
    df = df.loc[:, ("detid","lat", "long")]

    # Remove missing geocoordinates and sensor ids
    df.dropna(subset=['lat', 'long', 'detid'], how='any', inplace=True)

    num_processes = min(500, len(df))
    df_chunks = divide_dataframe(df, num_processes)

    # Create the networkx graph
    # 'drive', 'bike', 'walk'
    graph = create_graph("Munich, Bavaria, Germany", "bike")

    # Create a process for each chunk of the df dataframe
    processes = []
    for i, df_chunk in enumerate(df_chunks):
        p = mp.Process(target=generate_adjacency_matrix, args=(df, df_chunk, i+1, graph))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Combine the CSV files
    combine_csv_files(df)

    # Transpose the combined adjacency matrix and save it as a CSV file
    adjacency_matrix_file = os.path.join(OS_PATH, '/output/munich/munich_adjacency_matrix.csv')
    transpose_adjacency_matrix(adjacency_matrix_file)

    transposed_matrix_file = os.path.join(OS_PATH, '/output/munich/munich_adjacency_matrix_transposed.csv')
    normalize_matrix(transposed_matrix_file)
