# Q: Was sind die Öffnungszeiten des Stimmlokals im {{location_desc}}?
# Relevant datasets: ['Stimmlokale']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("location_desc", type=str)
args = parser.parse_args()
# Layers in dataset Stimmlokale: ['stzh.poi_stimmlokal_att', 'stzh.poi_stimmlokal_view']

### Solution ###
gdf_stimm = gpd.read_file("./data/opendata/50000006/extracted/stimmlokale.gpkg", layer="stzh.poi_stimmlokal_view")

lokal = gdf_stimm[gdf_stimm["name"].str.contains(args.location_desc.strip(), case=False, na=False)]
print(lokal["oeffnung"].values[0])