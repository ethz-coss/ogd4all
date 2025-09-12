# Q: Sind die Velo Abstellplätze mit Objekt-ID {{objectid_tuple}} im gleichen Kleinquartier (Kleinquartiere Stand 2024)?
# Relevant datasets: ['Kleinquartiere', 'Zweiradparkierung']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("objectid_tuple", type=str)
args = parser.parse_args()
# Layers in dataset Kleinquartiere: ['afs.kleinquartiere_2020', 'afs.kleinquartiere_2021', 'afs.kleinquartiere_2023', 'afs.kleinquartiere_2019', 'afs.kleinquartiere_2022', 'afs.kleinquartiere_2024']
# Layers in dataset Zweiradparkierung: ['taz.zweiradabstellplaetze_p']

### Solution ###
gdf_klein = gpd.read_file("./data/opendata/50000006/extracted/kleinquartiere.gpkg", layer="afs.kleinquartiere_2024")
gdf_zweir = gpd.read_file("./data/opendata/50000006/extracted/zweiradparkierung.gpkg", layer="taz.zweiradabstellplaetze_p")

velo1, velo2 = args.objectid_tuple.split(" und ")
velo1_geom = gdf_zweir[gdf_zweir["objectid"] == int(velo1)].geometry.iloc[0]
velo2_geom = gdf_zweir[gdf_zweir["objectid"] == int(velo2)].geometry.iloc[0]

velo1_quartier = gdf_klein[gdf_klein.contains(velo1_geom)]
velo2_quartier = gdf_klein[gdf_klein.contains(velo2_geom)]
same = velo1_quartier.iloc[0]["objectid"] == velo2_quartier.iloc[0]["objectid"]
print("Ja" if same else "Nein")