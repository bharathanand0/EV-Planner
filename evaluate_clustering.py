import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import osrm
import sklearn

def evaluate_clustering(model_filename, points_filename, weights_filename, plot_cdf=False, evaluate_existing=False, metrics = ["haversine"]):
    #model_filename = "models\\model_kmeans.pickle"
    #points_filename = "points_NY_popdist.csv"
    #weights_filename = "weights_NY_popdist.csv"

    def distance(point1, point2, metric = "haversine", osrm_matrix = [], point1_index = 0, point2_index = 0):
        if metric == "haversine":
            # approximate radius of earth in km
            R = 6371.0

            lat1 = radians(point1[0])
            lon1 = radians(point1[1])
            lat2 = radians(point2[0])
            lon2 = radians(point2[1])

            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))

            distance = R * c
            return distance
        elif metric =="euclidean":
            sum = 0
            for i in range(len(point1)):
                sum += (point1[i] - point2[i])**2
            return(sum**0.5)
        elif metric == "osrm-distance":
            return osrm_matrix[int(point1_index)][int(point2_index)]/1000.0
        elif metric == "osrm-time":
            return osrm_matrix[int(point1_index)][int(point2_index)]/60.0
        else:
            print("ERROR: Unsupported distance metric " + metric)
            quit()

    def min_dist_euclidean(point, centers):
        min_dist = distance(point, centers[0], "euclidean")
        label = 0
        for i in range(1, len(centers)):
            dst = distance(point, centers[i], "euclidean")
            if dst < min_dist:
                min_dist = dst
                label = i
        return(label)

    points = np.loadtxt(points_filename, delimiter=",")
    weights = np.loadtxt(weights_filename, delimiter=",")

    if(evaluate_existing):
        curr_stations_filename = model_filename
        df = pd.read_csv(curr_stations_filename)#######
        curr_stations = np.array(df[["Latitude", "Longitude"]])
        centers = curr_stations
        distances = sklearn.metrics.pairwise_distances(points,centers)
        #print("Distances between points and centers shape", distances.shape)
        labels = np.zeros(len(points), dtype = np.int32)
        for i in range(len(points)):
            labels[i] = np.argmin(distances[i])
        #print(labels)
        #labels = np.array([min_dist_euclidean(point, centers) for point in points])
    else:
        model = pickle.load(open(model_filename, 'rb'))
        centers = model.cluster_centers_
        labels = model.labels_

    #print("inertia: ", end = "")
    #print(centers[:10])
    #print(len(centers))
    #print(labels[:10])
    #print(len(labels))
    all_dists = []
    for metric in metrics:
        dists=[]
        sum_dists = 0
        sum_weights = 0
        inertia = 0

        if(metric == "osrm-distance"):
            annotation = "distance"
            #osrm_matrix = osrm.table(coords_src=points.tolist(), coords_dest=centers.tolist(), annotations = "distance")[0]
        elif(metric == "osrm-time"):
            annotation = "duration"
            #osrm_matrix = osrm.table(coords_src=points.tolist(), coords_dest=centers.tolist(), annotations = "duration")[0]
        else:
            annotation = "none"

        if annotation == "none":
            osrm_matrix=[]
        else:
            srcs = np.flip(points, axis = 1).tolist()
            num_srcs = len(srcs)
            dests = np.flip(centers, axis = 1).tolist()
            osrm_matrix = np.empty((num_srcs, len(dests)), dtype = np.float32)
            chunk_size = 1000
            for i in range(0, num_srcs, chunk_size):
                osrm_matrix[i:min(num_srcs, i+chunk_size)] = osrm.table(coords_src = srcs[i:min(num_srcs, i+chunk_size)], coords_dest=dests, annotations = annotation)[0]
        #np.savetxt("C:\\Users\\raghu\\Downloads\\poop.csv", osrm_matrix, delimiter=",")
        weights_mask = np.full(len(weights),True)
        for i in range(len(points)):
            dist_curr = distance(points[i],centers[int(labels[i])], metric, osrm_matrix, i, labels[i])
            if np.isnan(dist_curr):
                weights_mask[i] = False
                continue
            dists.append(dist_curr)
            sum_dists += dist_curr*weights[i]
            inertia += (dist_curr**2)*weights[i]
            sum_weights += weights[i]
        #print(dists[:10])
        #print(len(dists))
        #print(sum_weights)
        #print("average euclidean dist: ", end = "")
        avg_dist = sum_dists/sum_weights
        filtered_weights = weights[weights_mask]
        print("Weights length: " + str(len(filtered_weights)))
        print("Distances length: " + str(len(dists)))
        hist, bin_edges = np.histogram(dists, weights = filtered_weights, density=False, bins=200)
        dist_cdf = np.cumsum(hist)
        dist_cdf /= dist_cdf[-1]
        #print(hist)
        #print(cdf)
        dist_bin_centers = [(bin_edges[x]+bin_edges[x+1])/2 for x in range(len(bin_edges)-1)]
        if plot_cdf:
            plt.plot(dist_bin_centers, dist_cdf, color='r')
            #plt.hist(dists, bins = 200, cumulative=True, label='CDF', histtype='step', color='k')
            plt.show()

        k = len(centers)
        counts = np.zeros(k)
        for i in range(len(labels)):
            counts[int(labels[i])] += weights[i]

        count_hist, bin_edges = np.histogram(counts, bins=k//4)
        count_bin_centers = [(bin_edges[x]+bin_edges[x+1])/2 for x in range(len(bin_edges)-1)]


        # Convert the numpy array to a nested list
        centers_list = centers.tolist()
        all_dists.append(avg_dist)

    return([inertia, all_dists, dist_bin_centers, dist_cdf, count_bin_centers, count_hist, centers_list])
#####eval = evaluate_clustering("models\\NY_model_kmeans_1000.pickle", "data\\NY_points_popdist.csv", "data\\NY_weights_popdist.csv", plot_cdf=True, eval_existing = False)
#####print(eval)
