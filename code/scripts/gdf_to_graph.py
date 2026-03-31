def gdfs_to_graph(points, connections, node_id_col): # TO DO: graph_attr as input
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
            # 
            #geometry=row.geometry,
        )

        for i, row in connections.iterrows():
            from_node, to_node = row['Fra_full'], row['Til_full']
            dist = row['Afstand (meter)']
            G.add_edge(from_node, to_node, weight=dist)
            # TO DO: Do it one more time but reversed, to get both directions
            # TO DO: add OD values as edge attributes
            G.add_edge(to_node, from_node, weight=dist)

    return G