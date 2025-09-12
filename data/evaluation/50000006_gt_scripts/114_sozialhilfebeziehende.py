# Q: In welchem Zürcher Stadtquartier war {{year}} die Sozialhilfequote am höchsten, und wie hoch ist sie (runde auf zwei Nachkommastellen)?
# Relevant datasets: ['Sozialhilfebeziehende der Stadt Zürich nach Stadtquartier']

import geopandas as gpd
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("year", type=str)
args = parser.parse_args()
# Layers in dataset Sozialhilfebeziehende der Stadt Zürich nach Stadtquartier: ['sd_sod_sozialhilfequote_stadtquartier']

### Solution ###
# Challenge: SH_Quote is not numeric if loaded with geopandas
gdf_sd_so = gpd.read_file("./data/opendata/50000006/extracted/sd_sod_sozialhilfequote_stadtquartier.csv", layer="sd_sod_sozialhilfequote_stadtquartier", encoding="utf-8")
gdf_filtered = gdf_sd_so[gdf_sd_so["Jahr"] == args.year.strip()]
gdf_filtered.loc[:, 'SH_Quote'] = pd.to_numeric(gdf_filtered['SH_Quote'], errors='coerce')
highest_row = gdf_filtered.loc[gdf_filtered["SH_Quote"].idxmax()]
print(f"{highest_row['Raum']} mit {round(float(highest_row['SH_Quote']), 2)} %")