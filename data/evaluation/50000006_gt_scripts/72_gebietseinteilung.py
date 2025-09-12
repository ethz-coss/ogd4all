# Q: Was ist die flächenmässig {{sort}} römisch-katholische Kirchgemeinde in der Stadt Zürich? Was ist die Fläche in Quadratkilometern, gerundet auf eine Nachkommastelle?
# Relevant datasets: ['Gebietseinteilung der römisch-katholischen Kirchgemeinden']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("sort", type=str)
args = parser.parse_args()
# Layers in dataset Gebietseinteilung der römisch-katholischen Kirchgemeinden: ['kirchezh.kirchgemeinde_roem_kath_a']

### Solution ###
if args.sort.strip() == "grösste":
    ascending = False
elif args.sort.strip() == "kleinste":
    ascending = True
else:
    raise ValueError("Invalid sort argument. Use 'grösste' or 'kleinste'.")

gdf_gebie = gpd.read_file("./data/opendata/50000006/extracted/gebietseinteilung_der_roemisch_katholischen_kirchgemeinden.gpkg", layer="kirchezh.kirchgemeinde_roem_kath_a")
gdf_gebie["area"] = gdf_gebie.geometry.to_crs(epsg=2056).area

sorted_gdf = gdf_gebie.sort_values(by="area", ascending=ascending)
print(f"{sorted_gdf.iloc[0]["bezeichnung"]} mit {round(sorted_gdf.iloc[0]['area'] / 1e6, 1)} km²")