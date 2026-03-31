import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import math


def save_gdf(gdf, filename, layer):
    """
    Save a GeoDataFrame to a geopackage.
    """

    filename_degrees = str(filename).replace(".gpkg", "_degrees.gpkg")
    filename_metric = str(filename).replace(".gpkg", "_metric.gpkg")

    if gdf.crs == "EPSG:4326":
        gdf_metric = gdf.to_crs(epsg=25832)
        gdf.to_file(filename_degrees, driver="GPKG", layer=layer)
        gdf_metric.to_file(filename_metric, driver="GPKG", layer=layer)
        return gdf, gdf_metric
    
    elif gdf.crs == "EPSG:25832":
        gdf_degrees = gdf.to_crs(epsg=4326)
        gdf.to_file(filename_metric, driver="GPKG", layer=layer)
        gdf_degrees.to_file(filename_degrees, driver="GPKG", layer=layer)
        return gdf_degrees, gdf
    
    else:
        raise ValueError("Unsupported CRS. Only EPSG:4326 and EPSG:25832 are supported")


def plot_columns(gdf, col_list, save_fig=[False, None], title_dic = None, cmap='viridis'):
    """
    Plot multiple columns of a GeoDataFrame in subplots.
    col_list: list of column names to plot
    save_fig: [bool, filename] - if True, save the figure as the given filename
    """

    num_plots = len(col_list)

    # max 3 columns
    n_cols = min(3, num_plots)

    # ceiling division to avoid extra empty rows
    n_rows = math.ceil(num_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows)) # 3 columns

    # --- Ensure axes is always iterable ---
    if isinstance(axes, plt.Axes):       # only 1 subplot
        axes = [axes]
    else:
        axes = axes.flatten()

    # --- Plot each column ---
    for i, col in enumerate(col_list):
        gdf.plot(column=col, ax=axes[i], legend=True, cmap=cmap, markersize=50)

        axes[i].set_title(title_dic.get(col, col) if title_dic else col)
        axes[i].set_axis_off()

    # delete unsused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_fig[0]:
        plt.savefig(f'../results/{save_fig[1]}.png')
    plt.show()


def area_overlap_w_neighborhood(neigh_gdf, area_gdf, aspect, aspect_group, source, save=True):
    """
    Calculating overlapping area size for each station...
    1. associate polygons with neighborhoods. E.g. a polygon can be associated with multiple neighborhoods if these overlap. One row for each building intersecting with any metro neighborhood.
    2. For every polygon in neigh_test, find where it overlaps with polygons in area_test.
    3. Calculate area of intersecting overlays
    4. Sum intersecting overlays per neighborhood id and area category (agg)
    5. get total area over all categories per neighborhood (totals)
    """
    #area_gdf = area_gdf[area_gdf.geom_type=='Polygon']
    area_gdf = area_gdf[area_gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]

    area_inter = gpd.overlay(neigh_gdf, area_gdf, how="intersection")
    area_inter['inter_m2'] = area_inter.geometry.area

    # sum overlapping area m2 for each neighborhood and area group
    agg = (area_inter
        .groupby(['id', aspect_group])['inter_m2']
        .sum()
        .reset_index()
        )
    
    aspect_group_col = agg[aspect_group].unique()

    # get total overlapping area per neighborhood
    totals = agg.groupby('id')['inter_m2'].sum().reset_index().rename(columns={'inter_m2':f'{aspect}_total_m2'})
    pivot = agg.pivot(index='id', columns=aspect_group, values='inter_m2').fillna(0)
    pivot = pivot.reset_index()

    # Merge totals back
    pivot = pivot.merge(totals, on='id')
    neigh_with_areashares = neigh_gdf.merge(pivot, on='id', how='left')

    # get percentage columns
    neigh_with_areashares_pct = neigh_with_areashares.copy()
    neigh_with_areashares_pct[f'{aspect}_total_m2'] = (neigh_with_areashares_pct[f'{aspect}_total_m2']/neigh_with_areashares_pct['area']) * 100
    
    for col in aspect_group_col:
        # for each calculate percentage compared to total neighborhood area
        neigh_with_areashares_pct[col] = (neigh_with_areashares_pct[col]/neigh_with_areashares_pct['area']) * 100

    if save: # TO DO: Test that this works
        area_inter.to_file("../data/processed/station_neighborhoods.gpkg", layer=aspect, driver="GPKG")
        print(f"Overlaying area {aspect} saved as geopackage.")

        gdf_save = neigh_with_areashares_pct.copy()
        gdf_save = gdf_save.drop(columns=['total_overlap_area', 'geometry', 'area'], axis=1)
        gdf_save.to_csv(f'../data/{source}/processed/station_{aspect}.csv', index=False)
        print(f"Neighborhood with {aspect} attributes saved as csv.")
    else:
        pass

    return area_inter, neigh_with_areashares, neigh_with_areashares_pct
