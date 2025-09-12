# Q: Was waren die zwei statistischen Quartiere mit dem höchsten Ausländeranteil in {{year}}, und grenzen die Quartiere aneinander?
# Relevant datasets: ['Statistische Quartiere', 'Bevölkerung nach Stadtquartier, Herkunft, Geschlecht und Alter']

import geopandas as gpd
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("year", type=str)
args = parser.parse_args()
# Layers in dataset Statistische Quartiere: ['stzh.adm_statistische_quartiere_map', 'stzh.adm_statistische_quartiere_b_p', 'stzh.adm_statistische_quartiere_v']
# Layers in dataset Bevölkerung nach Stadtquartier, Herkunft, Geschlecht und Alter: ['bev390od3903']

### Solution ###
gdf_stati = gpd.read_file("./data/opendata/50000006/extracted/statistische_quartiere.gpkg", layer="stzh.adm_statistische_quartiere_v")
gdf_bev39 = gpd.read_file("./data/opendata/50000006/extracted/bev390od3903.csv", layer="bev390od3903", encoding="utf-8")

gdf_bev39["AnzBestWir"] = pd.to_numeric(gdf_bev39["AnzBestWir"]).fillna(0).astype(int)

grouped = (
    gdf_bev39[gdf_bev39["StichtagDatJahr"] == args.year.strip()]
    .groupby(["QuarLang", "HerkunftLang"], as_index=False)["AnzBestWir"]
    .sum()
    .pivot(index="QuarLang", columns="HerkunftLang", values="AnzBestWir")
    .fillna(0)
    .reset_index()
)
grouped["Ausländeranteil"] = grouped["Ausländer*in"] / (grouped["Ausländer*in"] + grouped["Schweizer*in"]) * 100
top2 = grouped.nlargest(2, "Ausländeranteil")

quartiere = gdf_stati[gdf_stati["qname"].isin(top2["QuarLang"])]
neighboring = quartiere.iloc[0].geometry.touches(quartiere.iloc[1].geometry)
touching = "ja" if neighboring else "nein"
print(f"{top2['QuarLang'].values[0]} und {top2['QuarLang'].values[1]}, {touching}")