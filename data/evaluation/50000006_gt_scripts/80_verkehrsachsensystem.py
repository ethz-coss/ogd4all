# Q: Auf wievielen Kilometern der Zürcher Strassen gilt {{tempo_regime}}. Runde auf eine Nachkommastelle.
# Relevant datasets: ['Verkehrsachsensystem Stadt Zuerich']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("tempo_regime", type=str)
args = parser.parse_args()
# Layers in dataset Verkehrsachsensystem Stadt Zuerich: ['vas.vas_e_strassenlaermemission', 'vas.vas_e_etappe3_strb', 'vas.vas_e_tempo_ist', 'vas.vas_e_verkehrstraeger', 'vas.vas_e_strassenbelag', 'vas.vas_e_rettungsachsen', 'vas.vas_e_steigung', 'vas.vas_e_kunstbauten', 'vas.vas_e_einbahn_ist', 'vas.vas_tempo_ist_event', 'vas.vas_basis', 'vas.vas_strassenbelag_event', 'vas.vas_strassenlaermemissio_event', 'vas.vas_etappe3_strb_event', 'vas.vas_verkehrstraeger_event', 'vas.vas_rettungsachsen_event', 'vas.vas_route', 'vas.vas_einbahn_ist_event', 'vas.vas_kunstbauten_event', 'vas.vas_steigung_event']

### Solution ###
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
gdf_hunde = gpd.read_file("./data/opendata/50000006/extracted/hundezonen.gpkg", layer="gsz.hundezone")
gdf_verke = gpd.read_file("./data/opendata/50000006/extracted/verkehrsachsensystem_stadt_zuerich.gpkg", layer="vas.vas_tempo_ist_event")

if args.tempo_regime.strip() == "Tempo-80":
    tempo_regime = "T80"
elif args.tempo_regime.strip() == "am Tag Tempo-50 und in der Nacht Tempo-30":
    tempo_regime = "T50 tagsüber, T30 nachts"
elif args.tempo_regime.strip() == "Fahrverbot (am Tag)":
    tempo_regime = "Fahrverbot"
else:
    raise ValueError(f"Unbekanntes Temporegime: {args.tempo_regime.strip()}")

gdf_filtered = gdf_verke[gdf_verke["temporegime"] == tempo_regime]
length_km = gdf_filtered.geometry.length.sum() / 1000 # m -> km
print(f"{round(length_km, 1)} km")