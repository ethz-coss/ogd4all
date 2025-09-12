# Q: Wieviele Quadratkilometer sind aktuell durch {{area_type}} bedeckt in der Stadt Zürich? Runde auf eine Nachkommastelle.
# Relevant datasets: ['Biotoptypenkartierung']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("area_type", type=str)
args = parser.parse_args()
# Layers in dataset Biotoptypenkartierung: ['gsz.btk']

### Solution ###
gdf_bioto = gpd.read_file("./data/opendata/50000006/extracted/biotoptypenkartierung.gpkg", layer="gsz.btk")

# lrtyp1text can also be used and leads to identical results
area = gdf_bioto[gdf_bioto["legende_lebensraeume"] == args.area_type.strip()]["geometry"].to_crs(epsg=2056).area.sum() / 1e6
print(f"{round(area, 1)} km²")