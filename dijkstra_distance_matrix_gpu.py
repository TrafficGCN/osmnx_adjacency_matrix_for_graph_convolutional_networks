import osmnx as ox
import networkx as nx
import pandas as pd
import cudf
import cugraph as cg
import os

# Function definitions
def fetch_road_network(place, network_type='drive'):
    ox.config(use_cache=True)
    graph = ox.graph_from_place(place, network_type=network_type)
    graph = ox.utils_graph.get_undirected(graph)
    return graph

def convert_nx_to_cugraph(nx_graph):
    edges_df = nx.to_pandas_edgelist(nx_graph)
    edges_df = edges_df.rename(columns={"source": "src", "target": "dst", "weight": "length"})
    edges_gdf = cudf.DataFrame.from_pandas(edges_df[['src', 'dst', 'length']])
    G = cg.Graph()
    G.from_cudf_edgelist(edges_gdf, source='src', destination='dst', edge_attr='length')
    return G

def find_nearest_nodes_and_calculate_shortest_paths(G, sensors_df, nx_graph):
    # Initialize an empty DataFrame for distances
    distances_df = pd.DataFrame(index=sensors_df['detid'], columns=sensors_df['detid'], data=float('inf'))
    
    for index, sensor_row in sensors_df.iterrows():
        sensor_lat = sensor_row['lat']
        sensor_long = sensor_row['long']
        sensor_detid = sensor_row['detid']
        
        # Find nearest node in the NetworkX graph (CPU operation)
        nearest_node = ox.get_nearest_node(nx_graph, (sensor_lat, sensor_long))
        
        # Store nearest node id for reference
        sensors_df.at[index, 'node_id'] = nearest_node
        
        # Compute shortest paths from this node to all others (GPU operation)
        df_shortest_paths = cg.sssp(G, nearest_node)
        
        # Map shortest paths back to sensors based on nearest nodes
        for _, target_sensor_row in sensors_df.iterrows():
            target_nearest_node = target_sensor_row['node_id']
            target_detid = target_sensor_row['detid']
            if nearest_node != target_nearest_node:
                try:
                    path_distance = df_shortest_paths.query('vertex == @target_nearest_node')['distance'].values[0]
                    distances_df.at[sensor_detid, target_detid] = path_distance
                except Exception as e:
                    print(f"Error computing path from {sensor_detid} to {target_detid}: {e}")
    
    distances_df.replace(float('inf'), 999999, inplace=True)
    return distances_df

# Main process
if __name__ == '__main__':
    # Fetch and prepare the road network
    place = "Munich, Bavaria, Germany"
    nx_graph = fetch_road_network(place, network_type='drive')
    
    # Convert the NetworkX graph to a cuGraph graph
    cu_graph = convert_nx_to_cugraph(nx_graph)
    
    # Data import path
    OS_PATH = os.path.dirname(os.path.realpath('__file__'))
    SENSORS_CSV = os.path.join(OS_PATH, 'data/munich/munich_sensors.csv')
    
    # Import sensors data
    sensors_df = pd.read_csv(SENSORS_CSV)
    sensors_df = sensors_df.loc[:, ["detid", "lat", "long"]]
    sensors_df.dropna(subset=['lat', 'long', 'detid'], how='any', inplace=True)
    
    # Find nearest nodes in the road network and calculate shortest paths using cuGraph
    adjacency_matrix = find_nearest_nodes_and_calculate_shortest_paths(cu_graph, sensors_df, nx_graph)
    
    # Output and save the adjacency matrix
    print(adjacency_matrix)
    adjacency_matrix.to_csv(os.path.join(OS_PATH, 'output/munich_adjacency_matrix.csv'))
