# Q: Wieviele {{parkplatz_typ}} gibt es insgesamt in öffentlichen Zürcher Parkhäusern?
# Relevant datasets: ['Öffentlich zugängliche Parkhäuser']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("parkplatz_typ", type=str)
args = parser.parse_args()
# Layers in dataset Öffentlich zugängliche Parkhäuser: ['stzh.poi_parkhaus_att', 'stzh.poi_parkhaus_view']

### Solution ###
gdf_oeffe = gpd.read_file("./data/opendata/50000006/extracted/oeffentlich_zugaengliche_parkhaeuser.gpkg", layer="stzh.poi_parkhaus_att")

if args.parkplatz_typ.strip() == "Behindertenparkplätze":
    num_pp = gdf_oeffe['davon_behinderten_pp'].sum()
elif args.parkplatz_typ.strip() == "Elektroparkplätze":
    num_pp = gdf_oeffe['davon_elektro_pp'].sum()
else:
    raise ValueError(f"Unknown parkplatz_typ: {args.parkplatz_typ}")
print(int(num_pp))