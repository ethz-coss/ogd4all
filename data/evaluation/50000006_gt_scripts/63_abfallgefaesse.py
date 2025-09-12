# Q: Wieviele {{gefaess_typ_txt}} gibt es in Zürich?
# Relevant datasets: ['Abfallgefässe']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("gefaess_typ_txt", type=str)
args = parser.parse_args()

### Solution ###
gdf_abfall = gpd.read_file("./data/opendata/50000006/extracted/abfallgefaesse.gpkg", layer="erz.abfallgefaess_p")

if args.gefaess_typ_txt.strip() == "Abfallhaie":
    gefaess_typ = 5
elif args.gefaess_typ_txt.strip() == "Robidogs":
    gefaess_typ = 0
elif args.gefaess_typ_txt.strip() == "Zürikübel":
    gefaess_typ = 4
else:
    raise ValueError(f"Unknown gefaess_typ_txt: {args.gefaess_typ_txt}")

gefaess_count = gdf_abfall[gdf_abfall["gefaesstyp"] == gefaess_typ].shape[0] # using gefaesstyp_txt yields identical results
print(gefaess_count)