# Q: (Fachplanung Hitzeminderung) Wie viel Prozent der Gesamtfläche der als '{{hotspot_type}}' markierten Flächen überschneiden sich mit den Flächen, die als 'Hotspot Nachtsituation' markiert sind? Runde auf zwei Nachkommastellen.
# Relevant datasets: ['Fachplanung Hitzeminderung OGD']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("hotspot_type", type=str)
args = parser.parse_args()
# Layers in dataset Fachplanung Hitzeminderung OGD: ['gsz.fph_massnahmengebiete_ogd', 'gsz.fph_hotspots_ogd', 'gsz.fph_aufwertung_ogd']

### Solution ###
gdf_fachp = gpd.read_file("./data/opendata/50000006/extracted/fachplanung_hitzeminderung_ogd.gpkg", layer="gsz.fph_hotspots_ogd")

if args.hotspot_type.strip() == "Hotspot Tag":
    type = "Hotspot Tagsituation"
elif args.hotspot_type.strip() == "Hotspot Tag/Nacht":
    type = "Hotspot Tag/Nacht"
else:
    raise ValueError("Invalid hotspot type. Must be 'Hotspot Tag' or 'Hotspot Tag/Nacht'.")

gdf_chosen_hotspot = gdf_fachp[gdf_fachp["hotspot"] == type]
gdf_hotspot_nacht = gdf_fachp[gdf_fachp["hotspot"] == "Hotspot Nachtsituation"]

intersection = gdf_chosen_hotspot.overlay(gdf_hotspot_nacht, how="intersection")
area_chosen = gdf_chosen_hotspot.geometry.area.sum()
area_intersection = intersection.geometry.area.sum()

percent_overlap = (area_intersection / area_chosen) * 100
print(f"{round(percent_overlap, 2)} %")
