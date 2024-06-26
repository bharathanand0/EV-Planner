import geopandas as gpd
import gzip, shutil
import pandas as pd
import numpy as np


# Specify the path to your compressed GPKG file
#compressed_filename = r"C:\Users\Anand Raghunathan\Downloads\kontur_population_GB_20231101.gpkg.gz"

# Specify the path to the decompressed file
#decompressed_filename = r"C:\Users\Anand Raghunathan\Downloads\kontur_population_GB_20231101.gpkg"

# Decompress the file
#with gzip.open(compressed_filename, 'rb') as f_in:
#    with open(decompressed_filename, 'wb') as f_out:
#        shutil.copyfileobj(f_in, f_out)

# Specify the path to your GPKG file
filename = "C:\\Users\\raghu\\Dropbox\\home\\bharath\\2024_science_fair_project\\data\\kontur_population_US_20231101.gpkg"


# Read the file
gdf = gpd.read_file(filename)
print(gdf.crs)
#quit()
gdf = gdf.to_crs(epsg=4326)
# Get the centroids of the hexagons
centroids = gdf.geometry.centroid
pop = gdf["population"].to_numpy()
# Get the latitude and longitude of the centroids
latitudes = centroids.y.to_numpy()
longitudes = centroids.x.to_numpy()
pop_lat_long = list(zip(latitudes, longitudes, pop))
#states = ["NC", "NJ", "ME", "CT", "CO"]
states = ["IN"]
bounding_boxes = {"IN": [-88.1,37.7,-84.7,41.8],"NC": [-84.4,33.8,-75.4,36.6], "NJ": [-75.6,38.9,-73.8,41.4], "ME": [-71.1,43.0,-66.9,47.5], "CT": [-73.8,40.9,-71.7,42.1], "CO": [-109.1,36.9,-102.0,41.1]}
for state in states:
    print(state)
    long_left = bounding_boxes[state][0]
    long_right = bounding_boxes[state][2]
    lat_top = bounding_boxes[state][3]
    lat_bot = bounding_boxes[state][1]

    print("About to filter lat, long")

    '''i = 0
    pop_lat_long_ny = []
    for row in pop_lat_long:
        i += 1
        lat = float(row[0])
        long = float(row[1])
        if((lat <= lat_top) and (lat >= lat_bot) and (long >= long_left) and (long <= long_right)):
            pop_lat_long_ny.append(row)
    pop_lat_long_ny = np.array(pop_lat_long_ny)
    '''

    pop_lat_long_ny = np.array([row for row in pop_lat_long if ((float(row[0])<=lat_top and float(row[0])>= lat_bot) and (float(row[1])>=long_left and float(row[1])<= long_right))])

    print(pop_lat_long_ny[:10])
    np.savetxt("C:\\Users\\raghu\\Dropbox\\home\\bharath\\2024_science_fair_project\\data\\"+state+"_pop_density.csv", pop_lat_long_ny, delimiter=",")
quit()
pd.DataFrame(gdf.assign(geometry=gdf["geometry"].apply(lambda p: p.wkt))).to_csv(r"C:\Users\Anand Raghunathan\Dropbox\home\bharath\2024_science_fair_project\pop_density.csv")

# Print the latitude and longitude of the centroids of the first ten hexagons
print(gdf.head(10))
#for i in range(10):
#    print(f'Hexagon {i+1}: Latitude: {latitudes[i]}, Longitude: {longitudes[i]}')
