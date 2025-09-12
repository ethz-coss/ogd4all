# Q: Welche Strasse ist die längste in Zürich, und wie viele Kilometer lang ist sie (runde auf zwei Nachkommastellen)? Bitte ignoriere Autobahnen.
# Relevant datasets: ['Strassennamenverzeichnis']

import geopandas as gpd


# Layers in dataset Strassennamenverzeichnis: ['geoz.sv_str_verz', 'geoz.sv_snb_beschluesse', 'geoz.sv_str_lin']

### Solution ###
# Challenge: What does it mean to ignore highways? Note that this avoids the problem of having to do a more involved length definition.
gdf_stras = gpd.read_file("./data/opendata/50000006/extracted/strassennamenverzeichnis.gpkg", layer="geoz.sv_str_lin")

gdf_stras["length"] = gdf_stras.geometry.length
gdf_stras.sort_values(by="length", ascending=False, inplace=True)
gdf_stras = gdf_stras[~gdf_stras["str_name"].str.match(r"^A\d+")]
print(f"{gdf_stras.iloc[0]["str_name"]} ({round(gdf_stras.iloc[0]['length'] / 1000, 2)} km)")