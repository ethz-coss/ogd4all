# Q: Hat es im Stadtkreis mit dem tiefsten Median-Vermögen (Stand {{year}}) von Verheirateten weniger städtische Familiengärten als im Kreis mit dem höchsten Median-Vermögen? Wenn ja, wieviele mehr?
# Relevant datasets: ['Median-Vermögen steuerpflichtiger natürlicher Personen nach Jahr, Steuertarif und Stadtkreis.', 'Stadtkreise', 'Familiengärten']

import geopandas as gpd
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("year", type=str)
args = parser.parse_args()
# Layers in dataset Median-Vermögen steuerpflichtiger natürlicher Personen nach Jahr, Steuertarif und Stadtkreis: ['wir100od1008']
# Layers in dataset Stadtkreise: ['stzh.adm_stadtkreise_v', 'stzh.adm_stadtkreise_a', 'stzh.adm_stadtkreise_beschr_p']
# Layers in dataset Familiengärten: ['stzh.poi_familiengarten_att', 'stzh.poi_familiengarten_view']

### Solution ###
gdf_wir10 = pd.read_csv("./data/opendata/50000006/extracted/wir100od1008.csv")
gdf_famil = gpd.read_file("./data/opendata/50000006/extracted/familiengaerten.gpkg", layer="stzh.poi_familiengarten_view")
gdf_stadt = gpd.read_file("./data/opendata/50000006/extracted/stadtkreise.gpkg", layer="stzh.adm_stadtkreise_v")

gdf_wir10 = gdf_wir10[gdf_wir10["StichtagDatJahr"] == int(args.year)]
gdf_wir10 = gdf_wir10[gdf_wir10["SteuerTarifLang"] == "Verheiratetentarif"]
lowest_kreis = gdf_wir10.nsmallest(1, "SteuerVermoegen_p50")["KreisSort"].values[0]
highest_kreis = gdf_wir10.nlargest(1, "SteuerVermoegen_p50")["KreisSort"].values[0]

gdf_lowest = gdf_stadt[gdf_stadt["knr"] == lowest_kreis].union_all()
gdf_highest = gdf_stadt[gdf_stadt["knr"] == highest_kreis].union_all()

low_contained = gdf_famil[gdf_famil.within(gdf_lowest)]
high_contained = gdf_famil[gdf_famil.within(gdf_highest)]
low_count = len(low_contained)
high_count = len(high_contained)

if low_count < high_count:
    print(f"Ja, nämlich {high_count - low_count} weniger.")
else:
    print(f"Nein")

