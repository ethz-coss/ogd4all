# Q: Gab es im Stadtkreis mit dem tiefsten Median-Vermögen (Stand {{year}}) von Verheirateten weniger von der Stadt als 'gute Bauten' ausgezeichnete Objekte (über alle Jahre) als im Kreis mit dem höchsten Median-Vermögen? Wenn ja, wieviele weniger?
# Relevant datasets: ['Median-Vermögen steuerpflichtiger natürlicher Personen nach Jahr, Steuertarif und Stadtkreis.', 'Auszeichnung für gute Bauten', 'Stadtkreise']

import geopandas as gpd
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("year", type=str)
args = parser.parse_args()
# Layers in dataset Median-Vermögen steuerpflichtiger natürlicher Personen nach Jahr, Steuertarif und Stadtkreis: ['wir100od1008']
# Layers in dataset Auszeichnung für gute Bauten: ['stzh.poi_agb_objektinventar_att', 'afs.agb_objektinventar_a', 'stzh.poi_agb_objektinventar_view']
# Layers in dataset Stadtkreise: ['stzh.adm_stadtkreise_v', 'stzh.adm_stadtkreise_a', 'stzh.adm_stadtkreise_beschr_p']

### Solution ###
# Challenges: Many datasets and required operations
gdf_wir10 = pd.read_csv("./data/opendata/50000006/extracted/wir100od1008.csv")
gdf_ausze = gpd.read_file("./data/opendata/50000006/extracted/auszeichnung_fuer_gute_bauten.gpkg", layer="afs.agb_objektinventar_a")
gdf_stadt = gpd.read_file("./data/opendata/50000006/extracted/stadtkreise.gpkg", layer="stzh.adm_stadtkreise_v")

gdf_wir10 = gdf_wir10[gdf_wir10["StichtagDatJahr"] == int(args.year)]
gdf_wir10 = gdf_wir10[gdf_wir10["SteuerTarifLang"] == "Verheiratetentarif"]
lowest_kreis = gdf_wir10.nsmallest(1, "SteuerVermoegen_p50")["KreisSort"].values[0]
highest_kreis = gdf_wir10.nlargest(1, "SteuerVermoegen_p50")["KreisSort"].values[0]

# Filtering removed as otherwise question always has "No" as answer
# gdf_ausze["zeitperiode_start"] = gdf_ausze["zeitperiode"].str.split(" - ").str[0].astype(int)
# gdf_ausze["zeitperiode_end"] = gdf_ausze["zeitperiode"].str.split(" - ").str[1].astype(int)
# gdf_ausze = gdf_ausze[(gdf_ausze["zeitperiode_start"] <= int(args.year)) & (gdf_ausze["zeitperiode_end"] >= int(args.year))]

gdf_lowest = gdf_stadt[gdf_stadt["knr"] == lowest_kreis].union_all()
gdf_highest = gdf_stadt[gdf_stadt["knr"] == highest_kreis].union_all()

low_contained = gdf_ausze[gdf_ausze.within(gdf_lowest)]
high_contained = gdf_ausze[gdf_ausze.within(gdf_highest)]
low_count = len(low_contained)
high_count = len(high_contained)

if low_count < high_count:
    print(f"Ja, nämlich {high_count - low_count} weniger.")
else:
    print(f"Nein")