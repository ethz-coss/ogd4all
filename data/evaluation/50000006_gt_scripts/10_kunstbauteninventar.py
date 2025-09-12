# Q: Wie gross ist die Fläche der {{bridge_name}} in Quadratmetern (gerundet auf ganze Zahl)?
# Relevant datasets: ['Kunstbauteninventar']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("bridge_name", type=str)
args = parser.parse_args()
# Layers in dataset Kunstbauteninventar: ['taz.view_kuba_linien', 'taz.view_kuba_flaechen']

### Solution ###
# Challenge: has to figure out which layer to use, not clear from the descriptions
gdf_kunst = gpd.read_file("./data/opendata/50000006/extracted/kunstbauteninventar.gpkg", layer="taz.view_kuba_flaechen")

bridge = gdf_kunst[gdf_kunst['bw_name'] == args.bridge_name.strip()]
bridge_area = bridge['geometry'].area
print(round(bridge_area.iloc[0]))