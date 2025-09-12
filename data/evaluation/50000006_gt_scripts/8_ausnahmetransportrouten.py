# Q: Wieviele Kilometer für den Ausnahmetransport von Typ {{type_number}} sind geplant? Bitte runde auf zwei Nachkommastellen.
# Relevant datasets: ['Ausnahmetransportrouten']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("type_number", type=str)
args = parser.parse_args()
# Layers in dataset Ausnahmetransportrouten: ['taz.ausnahmetransportrouten']

### Solution ###
# Challenge: How to figure out which routes are planned.
gdf_ausna = gpd.read_file("./data/opendata/50000006/extracted/ausnahmetransportrouten.gpkg", layer="taz.ausnahmetransportrouten")

gdf_typ = gdf_ausna[gdf_ausna['typ'] == f'ATR Typ {args.type_number.strip()} geplant'].copy()
gdf_typ['length'] = gdf_typ['geometry'].length / 1000
print(round(gdf_typ['length'].sum(), 2))