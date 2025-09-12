# Q: Um was für einen Faktor hat es mehr Sitzbänke pro Fläche in Gebieten von {{bedeutung_lang}} Bedeutung, im Vergleich zu Gebieten nachbarschaftlicher Bedeutung? Runde auf zwei Nachkommastellen.
# Relevant datasets: ['Bedeutungsplan', 'Sitzbankkataster OGD']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("bedeutung_lang", type=str)
args = parser.parse_args()
# Layers in dataset Bedeutungsplan: ['taz.view_bedeutungsplan_aktuell']
# Layers in dataset Sitzbankkataster OGD: ['gsz.bankstandorte_ogd']

### Solution ###
if args.bedeutung_lang.strip() == "internationaler":
    code_bedeutung = 1
elif args.bedeutung_lang.strip() == "stadtweiter":
    code_bedeutung = 2
elif args.bedeutung_lang.strip() == "quartierweiter":
    code_bedeutung = 3
else:
    raise ValueError("Unbekannte Bedeutung: " + args.bedeutung_lang)

gdf_bedeu = gpd.read_file("./data/opendata/50000006/extracted/bedeutungsplan.gpkg", layer="taz.view_bedeutungsplan_aktuell")
gdf_sitzb = gpd.read_file("./data/opendata/50000006/extracted/sitzbankkataster_ogd.gpkg", layer="gsz.bankstandorte_ogd")

gdf_joined = gpd.sjoin(gdf_sitzb, gdf_bedeu[["bedeutung", "geometry"]], how="left", predicate="within")
num_sitzb_bedeutung = gdf_joined[gdf_joined["bedeutung"] == code_bedeutung].shape[0]
num_sitzb_nachbarschaftlich = gdf_joined[gdf_joined["bedeutung"] == 4].shape[0]

area_bedeutung = gdf_bedeu[gdf_bedeu["bedeutung"] == code_bedeutung].geometry.to_crs(epsg=2056).area.sum()
area_nachbarschaftlich = gdf_bedeu[gdf_bedeu["bedeutung"] == 4].geometry.to_crs(epsg=2056).area.sum()

sitzb_density_bedeutung = num_sitzb_bedeutung / (area_bedeutung)
sitzb_density_nachbarschaftlich = num_sitzb_nachbarschaftlich / (area_nachbarschaftlich)
factor = sitzb_density_bedeutung / sitzb_density_nachbarschaftlich

print(round(factor, 2))

# Alternative answer: Don't count overlapping areas
# area_bedeutung_dissolved = gdf_bedeu[gdf_bedeu["bedeutung"] == code_bedeutung].geometry.to_crs(epsg=2056).geometry.union_all().area
# area_nachbarschaftlich_dissolved = gdf_bedeu[gdf_bedeu["bedeutung"] == 4].geometry.to_crs(epsg=2056).geometry.union_all().area
# sitzb_density_bedeutung_dissolved = num_sitzb_bedeutung / (area_bedeutung_dissolved)
# sitzb_density_nachbarschaftlich_dissolved = num_sitzb_nachbarschaftlich / (area_nachbarschaftlich_dissolved)
# factor_dissolved = sitzb_density_bedeutung_dissolved / sitzb_density_nachbarschaftlich_dissolved

# print(round(factor_dissolved, 2))
