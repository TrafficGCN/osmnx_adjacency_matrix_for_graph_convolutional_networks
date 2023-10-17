# Creating an Adjacency Matrix Using the Dijkstra Algorithm for Graph Convolutional Networks GCNs

<b>For more datasets view: https://github.com/ThomasAFink/40_cities_osmnx_adjacency_matrices_for_graph_convolutional_networks/</b>

### Citations

1. For the Los Angeles metr-la and Santa Clara pems-bay datasets cite: Kwak, Semin. (2020). PEMS-BAY and METR-LA in csv [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5146275

2. Weather sensors are provided by University of Utah Department of Atmospheric Sciences https://mesowest.utah.edu/

3. Bicycle sensors from the City of Munich Opendata Portal: https://opendata.muenchen.de/dataset/raddauerzaehlstellen-muenchen/resource/211e882d-fadd-468a-bf8a-0014ae65a393?view_id=11a47d6c-0bc1-4bfa-93ea-126089b59c3d

4. OpenStreetMap https://www.openstreetmap.org/ must also be referenced because the matrices where calculated using OpenStreetMap.
5. If you use any of the Maps you must reference both OpenStreetMap and Mapbox https://www.mapbox.com/.

### Introduction

Alright so you have a list of geocoordinates (for example maybe street sensors) and you want to create an adjacency matrix to feed into your PyTorch or Tensorflow GCN model such as the A3T-GCN. This tutorial uses OSMnx to measure the distance of optimal paths between each of the geocoordinates for each geocoordinate. The distances are saved to a CSV file as an adjacency matrix. The distance matrix in this tutorial is very basic, but you could go even further by capturing more spatial information such as capturing the shape of the paths between the geocoordinates. The adjacency matrix is used to structure the unstructured graph data.

![image](https://user-images.githubusercontent.com/53316058/217098922-fb6fb157-20dd-443f-8faa-217122097361.png)

### Repo Structure
A brief file structure overview of the repository is provided. The adjacency_matrix.py is in the root directory. The data folder houses a list of geocoordinates in a csv file. For this example six bicycle sensors where selected from the City of Munich's opendata portal. For a GCN of course it is better for a graph to be larger than 50 sensors. The graph distances are saved as a matrix in the output folder.



    /
    dijkstra_distance_matrix.py
    dijkstra_distance_matrix_multi_processing.py

    - / data / munich /
    munich_bicycle_sensors.csv

    - / output / munich /
    bicycle_adjacency_matrix.csv
    
The munich_bicycle_sensors.csv includes the following target geocoordinates.

    DETEKTOR_ID,  LATITUDE,    LONGITUDE
    Arnulf,       48.14205,    11.55534
    Kreuther,     48.12194,    11.62417
    Olympia,      48.16887,    11.55005
    Hirsch,       48.14438,    11.51794
    Margareten,   48.12032,    11.53599
    Erhardt,      48.13192,    11.58469
    
### Prerequisites
Before jumping into the code the following requirements and packages are needed to run the code:

    Python 3.10.6
    pip3 install osmnx==0.16.1
    pip3 install shapely==1.8.0
    pip3 install scipy
    pip3 install networkx
    pip3 install pandas
    
First the packages that were just installed are imported into our file adjacency_matrix.py

    import os
    import numpy as np
    from itertools import islice
    import pandas as pd
    import osmnx as ox
    import networkx as nx

### Code
Then we setup the OSMNX cache configuration is an optional method to store maps from OSM. This is especially resourceful for storing larger maps as it requires fewer requests to the api. For plotting a handful of paths this really does not impact processing time significantly.

    # Using the cache accelerates processing for a large map
    ox.config(log_console=True, use_cache=True)

Next we can set the location of our map to confine its perimeter.

    place = 'Munich, Bavaria, Germany'

The mode of transport is set to bike. Drive captures the road network used by automobiles. Bike capture the path network used by bicycles. Walk capture the walkway network used by pedestrians. Finding an optimal walkway is usually too resource intensive when processing due to the fact that their are many more nodes on the walking graph

    # 'drive', 'bike', 'walk'
    mode = 'bike'

Finally the graph can be requested and downloaded from the OSMnx api. The graph perimeters are set in the parameters and passed to the api request. The graph mode and place are also passed along. ONLY DOWNLOAD THE GRAPH ONCE TO AVOID REQUEST LIMITS!!

    graph = ox.graph_from_place(place, network_type = mode)    

Next a function is created which saves our optimal path distance matrix as the CSV file in our output folder from the list of geocoordinates in our data folder. This function takes our dataframe as a parameter.

    ##### Interface to OSMNX    
    def generate_adjacency_matrix(df):
    
To save the distance between every sensor with every other sensor we create an adjacency matrix.

        # Create the adjacency matrix
        matrix = [["DETEKTOR_ID_X, DETEKTOR_ID_Y, DISTANCE"]]
    
We need to iterate through all the detectors for each detector in our dataframe.

        i=0;
            for detector in islice(df.iterrows(), 0, len(list(df.DETEKTOR_ID))):
                j=0;
                for each_detector in df.iterrows():

Now that we have requested the graph from OSMnx api we can also request or path. We need the coordinates from the current sensor and the coordinates to the destination sensor. Remember we are inside a double for loop looping through sensor by sensor for each sensor.

                    # coordinates from the current sensor
                    start_latlng = (float(detector["LATITUDE"]), float(detector["LONGITUDE"]))
                    # coordinates belonging to the destination sensor
                    end_latlng = (float(each_detector["LATITUDE"]), float(each_detector["LONGITUDE"]))


Then we find the nearest nodes to the coordinates on the graph we requested from OSMnx.

                    # find the nearest node to the current sensor
                    orig_node = ox.get_nearest_node(graph, start_latlng)
                    # find the nearest node to the destination sensor
                    dest_node = ox.get_nearest_node(graph, end_latlng)


Using the Dijkstra method we find the shortest distance between the two sensors or points. Another method is the bellman-ford algorithm.

                    #find the shortest path method dijkstra or bellman-ford
                    shortest_route_distance = nx.shortest_path_length(graph, orig_node,dest_node, weight="length", method="dijkstra")

Finally we then append the distance to the adjacency matrix we created outside of the for loop. This would be a good place to console log the matrix to keep track of progress.

                    matrix.append([str(detector["DETEKTOR_ID"]) + "," + str(each_detector["DETEKTOR_ID"]) + "," + str(float(shortest_route_distance))])


After the distances between all the sensors have been found and added to the adjacency matrix using the nested for loop, the matrix is then converted into a numpy array.

        matrix = np.array(matrix)
        matrix = np.asarray(matrix)

Finally the matrix is saved as a csv in the output folder as bicycle_adjacency_matrix.csv

        # Save the dijkstra rad sensors distance matrix
        np.savetxt(OS_PATH + "output/bicycle_adjacency_matrix.csv", matrix, delimiter=",", fmt='%s')


<hr />


First in the main part of the script a list of target geocoordinates are fetched from the data folder in the munich_bicycle_sensors.csv file and loaded into python using pandas' dataframe method. The geocoordinates are then formatted as geocoordinates using geopandas.

    # Data import path
    OS_PATH = os.path.dirname(os.path.realpath('__file__'))
    SENSORS_CSV   = OS_PATH + '/data/munich_bicycle_sensors.csv'

    # Data Import Path
    df = pd.read_csv(SENSORS_CSV)

    # Keep only relevant columns
    df = df.loc[:, ("DETEKTOR_ID","LATITUDE", "LONGITUDE")]

    # Remove missing geocoordinates
    df.dropna(subset=['LATITUDE'], how='all', inplace=True)
    df.dropna(subset=['LONGITUDE'], how='all', inplace=True)

    # Remove missing sensor ids
    df.dropna(subset=['DETEKTOR_ID'], how='all', inplace=True)

Finally our generate_adjacency_matrix function is called after getting the optimal paths.

    generate_adjacency_matrix(df)

The result is presented below:

    DETEKTOR_ID_X, DETEKTOR_ID_Y, DISTANCE
    Arnulf,        Arnulf,        0.0
    Arnulf,        Kreuther,      6360.067999999998
    Arnulf,        Olympia,       3548.8400000000006
    Arnulf,        Hirsch,        3364.7529999999997
    Arnulf,        Margareten,    3658.5860000000002
    Arnulf,        Erhardt,       2916.509
    Kreuther,      Arnulf,        6403.099000000001
    Kreuther,      Kreuther,      0.0
    Kreuther,      Olympia,       9011.475000000002
    Kreuther,      Hirsch,        9707.515000000003
    Kreuther,      Margareten,    8099.6449999999995
    Kreuther,      Erhardt,       3523.4769999999994
    Olympia,       Arnulf,        3542.883
    Olympia,       Kreuther,      9088.311999999993
    Olympia,       Olympia,       0.0
    Olympia,       Hirsch,        4825.775000000002
    Olympia,       Margareten,    7074.653000000002
    Olympia,       Erhardt,       5913.148000000002
    Hirsch,        Arnulf,        3302.0229999999992
    Hirsch,        Kreuther,      9655.762999999995
    Hirsch,        Olympia,       4833.914000000002
    Hirsch,        Hirsch,        0.0
    Hirsch,        Margareten,    4234.665
    Hirsch,        Erhardt,       6212.2040000000015
    Margareten,    Arnulf,        3646.659999999999
    Margareten,    Kreuther,      8178.820999999996
    Margareten,    Olympia,       6458.455999999998
    Margareten,    Hirsch,        4202.3730000000005
    Margareten,    Margareten,    0.0
    Margareten,    Erhardt,       4780.81
    Erhardt,       Arnulf,        2913.113999999999
    Erhardt,       Kreuther,      3536.422
    Erhardt,       Olympia,       5837.165999999999
    Erhardt,       Hirsch,        6217.530000000001
    Erhardt,       Margareten,    4748.426999999998
    Erhardt,       Erhardt,       0.0
<img src="https://github.com/ThomasAFink/osmnx_adjacency_matrix_for_graph_convolutional_networks/blob/main/output/munich/munich_dijkstra_map_bike_small.jpg?raw=true" width="450" align="right">

A more complex adjacency may capture the entire shape of the paths between the points instead of just the distance. This would increase the accuracy of a GCN model as you capturing more spatial information. This can also be done with OSMnx. I reference how to capture an entire path in an array plotting the Dijkstra paths in my other tutorial.

The OSMnx solution is great for data science if you need static graph data to test an academic GCN model. For a dynamic graph or a software solution it would be better to use a proprietary vendor such as a Google or Apple Maps to calculate your distances with better efficiency.
