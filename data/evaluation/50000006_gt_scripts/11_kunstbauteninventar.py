# Q: Wieviele {{bw_category}} hat es in der Stadt Zürich?
# Relevant datasets: ['Kunstbauteninventar']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("bw_category", type=str)
args = parser.parse_args()
# Layers in dataset Kunstbauteninventar: ['taz.view_kuba_linien', 'taz.view_kuba_flaechen']

### Solution ###
gdf_kunst = gpd.read_file("./data/opendata/50000006/extracted/kunstbauteninventar.gpkg", layer="taz.view_kuba_flaechen")

if args.bw_category == "Fussgängerunterführungen":
    gdf_kunst_filtered = gdf_kunst[gdf_kunst['kategorie'] == 'Fussgängerunterführung']
elif args.bw_category == "Strassenunterführungen":
    gdf_kunst_filtered = gdf_kunst[gdf_kunst['kategorie'] == 'Strassenunterführung']
elif args.bw_category == "Strassentunnel":
    gdf_kunst_filtered = gdf_kunst[gdf_kunst['kategorie'] == 'Strassentunnel']
else:
    raise ValueError(f"Unknown category: {args.bw_category}")
print(gdf_kunst_filtered.shape[0])