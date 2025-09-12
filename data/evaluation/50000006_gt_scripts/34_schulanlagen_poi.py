# Q: Welche Schulanlage ist am weitesten vom nächsten öffentlichen Spielplatz entfernt? Berücksichtige bei den Schulanlagen nur solche von 2024. Gib auch die Entfernung in Metern an (gerundet auf ganze Zahl).
# Relevant datasets: ['Schulanlagen', 'POI öffentliche Spielplätze']

import geopandas as gpd


# Layers in dataset Schulanlagen: ['stzh.poi_kinderhort_att', 'stzh.poi_volksschule_att', 'stzh.poi_kindergarten_att', 'stzh.poi_kindergarten_view', 'stzh.poi_kinderhort_view', 'stzh.poi_volksschule_view', 'ssd.schulanlagen_archiv', 'stzh.poi_sonderschule_view']
# Layers in dataset POI öffentliche Spielplätze: ['stzh.poi_spielplatz_att', 'stzh.poi_spielplatz_view']

### Solution ###
gdf_schul = gpd.read_file("./data/opendata/50000006/extracted/schulanlagen.gpkg", layer="ssd.schulanlagen_archiv")
gdf_poi_o = gpd.read_file("./data/opendata/50000006/extracted/poi_oeffentliche_spielplaetze.gpkg", layer="stzh.poi_spielplatz_view")

gdf_schul = gdf_schul[gdf_schul["jahresstand"] == 2024]
nearest_idx = gdf_schul.geometry.apply(lambda x: gdf_poi_o.distance(x).idxmin())
gdf_schul["dist_min"] = gdf_schul.geometry.apply(lambda x: gdf_poi_o.distance(x).min())
max_idx = gdf_schul["dist_min"].idxmax()
print(f"{gdf_schul.loc[max_idx]['typ']} {gdf_schul.loc[max_idx]['name']} (Distanz: {round(gdf_schul.loc[max_idx]['dist_min'])} m)")