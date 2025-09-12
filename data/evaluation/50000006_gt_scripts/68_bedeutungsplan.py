# Q: Wieviele Quadratmeter der Stadtfläche sind gemäss Bedeutungsplan von {{bedeutung_lang}} Bedeutung? Runde auf eine ganze Zahl.
# Relevant datasets: ['Bedeutungsplan']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("bedeutung_lang", type=str)
args = parser.parse_args()
# Layers in dataset Bedeutungsplan: ['taz.view_bedeutungsplan_aktuell']

### Solution ###
# 1 international 2 stadtweit 3 quartierweit 4 nachbarschaftlich
gdf_bedeu = gpd.read_file("./data/opendata/50000006/extracted/bedeutungsplan.gpkg", layer="taz.view_bedeutungsplan_aktuell")

if args.bedeutung_lang.strip() == "internationaler":
    code_bedeutung = 1
elif args.bedeutung_lang.strip() == "stadtweiter":
    code_bedeutung = 2
elif args.bedeutung_lang.strip() == "quartierweiter":
    code_bedeutung = 3
elif args.bedeutung_lang.strip() == "nachbarschaftlicher":
    code_bedeutung = 4
else:
    raise ValueError("Unbekannte Bedeutung: " + args.bedeutung_lang)

print(f"{round(gdf_bedeu[gdf_bedeu["bedeutung"] == code_bedeutung].geometry.to_crs(epsg=2056).area.sum())} m²")

# Alternative valid answer
# subset = gdf_bedeu[gdf_bedeu['bedeutung'] == code_bedeutung].to_crs(epsg=2056)
# dissolved_geom = subset.geometry.union_all()  # merges overlaps
# dissolved_area = round(gpd.GeoSeries([dissolved_geom], crs=subset.crs).area.iloc[0])
# print(f"{dissolved_area} m² (ohne Überschneidungen)")
