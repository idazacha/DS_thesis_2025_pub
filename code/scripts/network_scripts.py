import pandas as pd
import networkx as nx

def station_flow_metrics(DiG,
                         BaG, 
                         od_df, 
                         trips_cols=['trips', 'morning_trips', 'noon_trips', 'afternoon_trips', 'evening_trips', 'night_trips'], 
                         #flow_col=["flow", "flow_morning", "flow_noon", "flow_afternoon", "flow_evening", "flow_night"],
                         station_names=None):
    metrics_df = pd.DataFrame()

    for trip_col in trips_cols:
        if trip_col =="trips":
            flow_col = "flow"
        else:
            time_split = trip_col.split("_")[0]
            flow_col = "flow_" + time_split
        #print(trip_col, flow_col)

        # 1. start and end nodes
        origins = od_df.groupby("from")[trip_col].sum().reindex(DG.nodes(), fill_value=0)
        destinations = od_df.groupby("to")[trip_col].sum().reindex(DG.nodes(), fill_value=0)
        metrics_df[f'{trip_col}_origins'] = origins
        metrics_df[f'{trip_col}_destinations'] = destinations

        # 2. node inflow and outflow
        inflow = DiG.in_degree(weight=flow_col)     # People arriving at or through station
        outflow = DiG.out_degree(weight=flow_col)   # People departing at or through station
        inflow_values = {node: inflow[node] for node in DiG.nodes()}
        outflow_values = {node: outflow[node] for node in DiG.nodes()}
        metrics_df[f'{trip_col}_inflow'] = inflow_values
        metrics_df[f'{trip_col}_outflow'] = outflow_values

        # 3. node through-flow
        node_flow = {node: 0 for node in DiG.nodes()}
        for u, v, data in DiG.edges(data=True):
            node_flow[u] += data[flow_col]
            node_flow[v] += data[flow_col]
        metrics_df[f'{trip_col}_node_flow'] = node_flow

        # 3. Transfer Flow
        through_flow = {}
        for n in DiG.nodes():
            through_flow[n] = inflow[n] + outflow[n] - origins[n] - destinations[n]
        metrics_df[f'{trip_col}_through_flow'] = through_flow


    # 4. Structural centralities
    closeness = nx.closeness_centrality(BaG, distance="distance")
    betweenness = nx.betweenness_centrality(BaG, weight="distance")
    metrics_df['closeness'] = closeness
    metrics_df['betweenness'] = betweenness

    metrics_df = metrics_df.reset_index(names=['station'])
    return metrics_df

# Function to compute mixing for a single time slice
def compute_temporal_mixing(od, trip_col):
    """
    od: dataframe with columns cluster_from, cluster_to, trip_col
    trip_col: str, e.g. 'morning_trips'
    """
    # NOTE: Made into a function using ChatGPT. The code itself and it functionality is selfwritten...

    df = (
        od
        .groupby(["cluster_from", "cluster_to"], as_index=False)[trip_col]
        .sum()
        .rename(columns={trip_col: "trips"})
    )

    df["total_out"] = df.groupby("cluster_from")["trips"].transform("sum")
    df["P_obs"] = df["trips"] / df["total_out"]

    P = df.pivot(
        index="cluster_from",
        columns="cluster_to",
        values="P_obs"
    ).fillna(0)

    return P