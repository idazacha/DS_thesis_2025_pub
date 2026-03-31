def gdfs_to_Graph(points, connections, node_id_col): # TO DO: graph_attr as input
    # Undirected Graph
    G = nx.Graph() # TO DO: make Graph attributes, e.g. nx.Graph(day="Friday")

    # add stations as nodes
    for _, row in points.iterrows():
        #print(row[node_id])
        G.add_node(
            row[node_id_col],
            name=row.name,
            id=row.id,
            geometry=row.geometry,
            system=row.system
            # color=row.color # TO DO: Assign color corresponding to system/branch
            # TO DO: add more attributes, e.g. lines serving the station
            # TO DO: add station numbers as attributes
            # TO DO: add neighborhood characteristics as attributes
            # TO DO: add neighborhood classification as attributes
            #geometry=row.geometry,
        )

        for i, row in connections.iterrows():
            from_node, to_node = row['Fra_full'], row['Til_full']
            dist = row['Afstand (meter)']
            G.add_edge(from_node, to_node, weight=dist)
            # TO DO: Do it one more time but reversed, to get both directions
            # TO DO: add OD values as edge attributes
    return G

def gdfs_to_DiGraph(points, connections, node_id_col): # TO DO: graph_attr as input
    # Directed Graph
    G_directed = nx.DiGraph()

    # add stations as nodes
    for _, row in points.iterrows():
        #print(row[node_id])
        G_directed.add_node(
            row[node_id_col],
            name=row.name,
            id=row.id,
            geometry=row.geometry,
            system=row.system
            # color=row.color # TO DO: Assign color corresponding to system/branch
            # TO DO: add more attributes, e.g. lines serving the station
            # TO DO: add station numbers as attributes
            # TO DO: add neighborhood characteristics as attributes
            # TO DO: add neighborhood classification as attributes
            #geometry=row.geometry,
        )

        for i, row in connections.iterrows():
            from_node, to_node = row['Fra_full'], row['Til_full']
            dist = row['Afstand (meter)']
            G_directed.add_edge(from_node, to_node, weight=dist)
            # TO DO: Do it one more time but reversed, to get both directions
            # TO DO: add OD values as edge attributes
            G_directed.add_edge(from_node, to_node, weight=dist)
            G_directed.add_edge(to_node, from_node, weight=dist)

    return G_directed