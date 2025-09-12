# Q: Wieviele Quadratmeter designierte Abstellflächen bietet die Stadt Zürich für Mikromobilitätsanbieter zur Verfügung? Runde auf ganze Zahlen.
# Relevant datasets: ['Sharing Zonen']

import geopandas as gpd


# Layers in dataset Sharing Zonen: ['dav.parkverbotszonen', 'dav.fahrverbotszonen', 'dav.parkierzonen']

### Solution ###
gdf_shari = gpd.read_file("./data/opendata/50000006/extracted/sharing_zonen.gpkg", layer="dav.parkierzonen")
gdf_shari['area'] = gdf_shari.geometry.area
print(round(gdf_shari['area'].sum()))