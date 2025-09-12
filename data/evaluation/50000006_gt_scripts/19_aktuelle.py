# Q: An welchem Standort hat die Stadtverwaltung Zürich die meisten Überwachungskameras im Aussenbereich installiert?
# Relevant datasets: ['Aktuelle Auflistung von Videokameras der Stadtverwaltung Zürich']

import geopandas as gpd


# Layers in dataset Aktuelle Auflistung von Videokameras der Stadtverwaltung Zürich: ['stez.liste_videokameras_stadtverwal']

### Solution ###
gdf_aktue = gpd.read_file("./data/opendata/50000006/extracted/aktuelle_auflistung_von_videokameras_der_stadtverwaltung_zuerich.gpkg", layer="stez.liste_videokameras_stadtverwal")

gdf_aktue.sort_values(by="anzahl_kameras_aussen", ascending=False, inplace=True)
print(gdf_aktue.iloc[0]["standort_beschreibung"])