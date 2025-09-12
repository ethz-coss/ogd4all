# Q: Auf wieviel Prozent der Fläche der Stadt Zürich gilt der {{tariff_type}} für das Parkieren? Runde auf zwei Nachkommastellen.
# Relevant datasets: ['Gebietseinteilung Parkierungsgebühren']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("tariff_type", type=str)
args = parser.parse_args()
# Layers in dataset Gebietseinteilung Parkierungsgebühren: ['dav.tarifzonen']

### Solution ###
gdf_gebie = gpd.read_file("./data/opendata/50000006/extracted/gebietseinteilung_parkierungsgebuehren.gpkg", layer="dav.tarifzonen")

gdf_gebie['area'] = gdf_gebie.geometry.to_crs(epsg=2056).area
total_area = gdf_gebie['area'].sum()
tariff_area = gdf_gebie[gdf_gebie['tarifzone'] == f"{args.tariff_type.strip()}zone"]['area'].sum()
tariff_percentage = (tariff_area / total_area) * 100
print(round(tariff_percentage, 2))