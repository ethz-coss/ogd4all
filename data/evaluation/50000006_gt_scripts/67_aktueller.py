# Q: Wieviel Prozent der Hundebesitzer im Quartier {{quartier_lang}} waren im Jahr 2024 männlich? Runde auf zwei Nachkommastellen.
# Relevant datasets: ['Hundebestände der Stadt Zürich, seit 2014']

import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("quartier_lang", type=str)
args = parser.parse_args()
# Layers in dataset Hundebestände der Stadt Zürich, seit 2014: ['kul100od1001']

### Solution ###
gdf_kul10 = gpd.read_file("./data/opendata/50000006/extracted/kul100od1001.csv", layer="kul100od1001", encoding="utf-8")
gdf_kul10_filtered = gdf_kul10[gdf_kul10["StichtagDatJahr"] == "2024"]
gdf_kul10_filtered = gdf_kul10_filtered[gdf_kul10_filtered["QuarLang"] == args.quartier_lang.strip()]
gdf_kul10_filtered = gdf_kul10_filtered[['HID', 'SexLang']].drop_duplicates() # some people may have multiple dogs

# Get value counts of field SexLang
gender_counts = gdf_kul10_filtered["SexLang"].value_counts()
print(f"{round(gender_counts["männlich"] / (gender_counts["männlich"] + gender_counts["weiblich"]) * 100, 2)} %")