# Q: Was ist das Verhältnis (in Prozent) von Behindertenparkplätzen zu der Gesamtanzahl an öffentlichen Parkplätzen? Runde auf zwei Nachkommastellen.
# Relevant datasets: ['Öffentlich zugängliche Parkhäuser', 'Öffentlich zugängliche Strassenparkplätze OGD']

import geopandas as gpd

# Layers in dataset Öffentlich zugängliche Parkhäuser: ['stzh.poi_parkhaus_att', 'stzh.poi_parkhaus_view']
# Layers in dataset Öffentlich zugängliche Strassenparkplätze OGD: ['taz.view_pp_ogd']

### Solution ###
# Challenge: The dataset "Behindertenparkplätze" is actually not needed, as it is a subset of the dataset "Öffentlich zugängliche Strassenparkplätze OGD"
gdf_oeffe_parkhaus = gpd.read_file("./data/opendata/50000006/extracted/oeffentlich_zugaengliche_parkhaeuser.gpkg", layer="stzh.poi_parkhaus_att")
gdf_oeffe_strasse = gpd.read_file("./data/opendata/50000006/extracted/oeffentlich_zugaengliche_strassenparkplaetze_ogd.gpkg", layer="taz.view_pp_ogd")

num_disabled_parking_spaces_parkhaus = gdf_oeffe_parkhaus['davon_behinderten_pp'].sum()
total_parking_spaces_parkhaus = gdf_oeffe_parkhaus['anzahl_oeffentliche_pp'].sum()
num_disabled_parking_spaces_strasse = gdf_oeffe_strasse[gdf_oeffe_strasse['art'] == "Nur mit Geh-Behindertenausweis"].shape[0]
total_parking_spaces_strasse = gdf_oeffe_strasse.shape[0]
print(round((num_disabled_parking_spaces_parkhaus + num_disabled_parking_spaces_strasse) / (total_parking_spaces_parkhaus + total_parking_spaces_strasse) * 100, 2))