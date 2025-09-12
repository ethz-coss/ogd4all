# Q: Welche Strasse verbindet die Adressen {{addr_list_str}}?
# Relevant datasets: ['Strassennamenverzeichnis']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("addr_list_str", type=str)
args = parser.parse_args()
# Layers in dataset Strassennamenverzeichnis: ['geoz.sv_str_verz', 'geoz.sv_snb_beschluesse', 'geoz.sv_str_lin']

### Solution ###
# Challenge: Have to figure out that should check both directions
gdf_stras = gpd.read_file("./data/opendata/50000006/extracted/strassennamenverzeichnis.gpkg", layer="geoz.sv_str_lin")

# Split the input string into two addresses
addr_list = args.addr_list_str.split(" und ")
addr1 = addr_list[0].strip()
addr2 = addr_list[1].strip()

# Filter str_von or str_bis and print name
gdf_filtered = gdf_stras[
    (gdf_stras["str_von"].str.contains(addr1, na=False) &
    gdf_stras["str_bis"].str.contains(addr2, na=False)) |
    (gdf_stras["str_von"].str.contains(addr2, na=False) &
    gdf_stras["str_bis"].str.contains(addr1, na=False))
]
    
print(gdf_filtered["str_name"].values[0])