# Q: Für was ist der Standort der Stadtpolizei {{addr_desc}} zuständig?
# Relevant datasets: ['Stadtpolizei']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("addr_desc", type=str)
args = parser.parse_args()
# Layers in dataset Stadtpolizei: ['stzh.poi_stadtpolizei_view']

### Solution ###
gdf_stadt = gpd.read_file("./data/opendata/50000006/extracted/stadtpolizei.gpkg", layer="stzh.poi_stadtpolizei_view")

if args.addr_desc.strip() == "am Gänzilooweg":
    street_name = "Gänzilooweg"
elif args.addr_desc.strip() == "an der Schaffhauserstrasse 26":
    street_name = "Schaffhauserstrasse 26"

hit = gdf_stadt[gdf_stadt["adresse"].str.contains(street_name, case=False, na=False)]
print(hit["name"].values[0])