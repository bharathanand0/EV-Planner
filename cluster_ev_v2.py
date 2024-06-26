import pandas as pd
import numpy as np
import random
#from cluster import createClusters
#from point import makePointList
#from kmeans import kmeans
#import geopy
from sklearn.cluster import *
#from sklearn.cluster import DBSCAN
from itertools import islice
from k_means_constrained import KMeansConstrained
import pickle
#from simple_kmeans import K_Means
import sklearn
from sklearn_extra.cluster import KMedoids
import osrm



def create_clustering_dataset(ev_reg_data_filename, zipcode_to_lat_long_filename, pop_density_filename, points_filename, weights_filename):

    #ev_reg = pd.read_csv(ev_reg_data_filename, dtype = {'State': str, 'ZIP Code': int, 'Registration Date': str, 'Vehicle Make': str, 'Vehicle Model': str, 'Vehicle Model Year': int, 'Drivetrain Type': str, 'Vehicle GVWR Class': str, 'Vehicle Category': str, 'Vehicle Count': int, 'DMV Snapshot ID': int, 'DMV Snapshot (Date)': str, 'Latest DMV Snapshot Flag': bool}, encoding_errors='ignore', on_bad_lines='skip')
    ev_reg = pd.read_csv(ev_reg_data_filename, on_bad_lines='skip')
    filtered_ev_reg = ev_reg.loc[ev_reg["Latest DMV Snapshot Flag"] == True]
    print("Read EV registration datset with " + str(len(ev_reg)) + " entries")
    npy_ev_reg = filtered_ev_reg.to_numpy()
    print("Created npy_ev_reg numpy array")
    print(npy_ev_reg[:5])

    zipcode_to_lat_long = pd.read_csv(zipcode_to_lat_long_filename)
    npy_zipcode_to_lat_long = zipcode_to_lat_long.to_numpy()
    dict_zp_to_ll = {npy_zipcode_to_lat_long[i][0]: (npy_zipcode_to_lat_long[i][1], npy_zipcode_to_lat_long[i][2]) for i in range(len(npy_zipcode_to_lat_long))}
    print("Created dictionary from zipcode to (lat,long) with " + str(len(dict_zp_to_ll)) + " entries")

    zip_to_data = dict()
    for row in npy_ev_reg:
        if(row[1] == 'Error'):
            continue
        if int(row[1]) in zip_to_data:
            curr_count = zip_to_data[int(row[1])][0]
            zip_to_data[int(row[1])] = [int(row[9])+curr_count, []]
        else:
            zip_to_data[int(row[1])] = [int(row[9]), []]

    print(dict(islice(zip_to_data.items(), 0, 10)))
    ev_count = 0

## DEBUG
#    for key, value in zip_to_data.items():
#        ev_count += value[0]
#    print("Total EV count: " + str(ev_count))
#    quit()

    lat_long_pop = np.loadtxt(pop_density_filename, delimiter=',')
    print(lat_long_pop[:10])

    def find_closest_zip(lat_target, long_target):
        min_dist = 79854753489453243
        min_zip = 0
        for [zipcode, lat, long] in npy_zipcode_to_lat_long:
            dist = ((lat - lat_target)**2 + (long - long_target)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                min_zip = zipcode
        return min_zip

    for [lat, long, pop] in lat_long_pop:
        zipcode = find_closest_zip(lat, long)
        try:
            zip_to_data[int(zipcode)][1].append([lat, long, pop])
        except:
            continue

    print(dict(islice(zip_to_data.items(), 0, 10)))

    points = []
    weights = []
    for key, value in zip_to_data.items():
        ev_count = value[0]
        lat_long_pop_lst = value[1]
        tot_pop = sum([x[2] for x in lat_long_pop_lst])
        for [lat, long, pop] in lat_long_pop_lst:
            points.append([lat, long])
            weights.append((1.0*ev_count*pop)/tot_pop)

    points_np = np.array(points)
    np.savetxt(points_filename, points_np, delimiter=",")
    #np.savetxt(r"C:\Users\Anand Raghunathan\Dropbox\home\bharath\2024_science_fair_project\points_NY_popdist.csv", points_np, delimiter=",")

    weights_np = np.array(weights)
    np.savetxt(weights_filename, weights_np, delimiter=",")

    #print("Created list with " + str(len(points)) + " points")

'''def find_n_closest_points(points_i, centroids, labels, k, n):
        # Filter points assigned to centroid k

        points_k = points_i[labels == k]

        # Calculate Euclidean distances
        distances = np.linalg.norm(points_k - centroids[k], axis=1)
        # Sort distances and get n closest indices
        closest_indices = np.argsort(distances)[:n]

        # Map back to original indices
        original_indices = np.where(labels == k)[0][closest_indices]
        #print(original_indices)
        return original_indices
        '''

def find_closest_points(points, centroids, labels, weights, distances, cluster_id, num_to_select):
    dists = distances[:,cluster_id][labels == cluster_id]
    weights_filtered = weights[labels == cluster_id]
    point_indices = np.argsort(dists)
    total_weight = 0.0
    i = 0
    for ind in point_indices:
        total_weight += weights_filtered[ind]
        i += 1
        if(total_weight > num_to_select):
            break

    point_indices = point_indices[:i]
    orig_indices = np.where(labels == cluster_id)[0][point_indices]
    return orig_indices

def osrm_dist(points, centers, annotation):
    srcs = np.flip(points, axis = 1).tolist()
    num_srcs = len(srcs)
    dests = np.flip(centers, axis = 1).tolist()
    osrm_matrix = np.empty((num_srcs, len(dests)), dtype = np.float32)
    chunk_size = 1000
    for i in range(0, num_srcs, chunk_size):
        osrm_matrix[i:min(num_srcs, i+chunk_size)] = osrm.table(coords_src = srcs[i:min(num_srcs, i+chunk_size)], coords_dest=dests, annotations = annotation)[0]
    return osrm_matrix
class SA_KMeans:
    def __init__(self, num_clusters, hyperparameters=[0.2, 0.8, 1.25], random_state = 101):
        self.num_clusters = num_clusters
        # Global copy of points, weights, labels and cluster centers
        self.points = None
        self.weights = None
        self.labels_ = None
        self.cluster_centers_ = np.zeros((num_clusters,2))

        #Points involved in current round of clustering and their indices into the global points array
        self.cur_points = None
        self.cur_points_indices = None
        self.cur_weights = None

        self.num_accepted_clusters = 0
        self.fraction_to_accept = hyperparameters[0] #Hyperparameter
        self.min_to_accept = 10
        self.random_state = random_state
        self.weight_dist = hyperparameters[1] #Hyperparameter
        self.weight_clust_size = 1 - hyperparameters[1] #Hyperparameter
        self.large_clust_threshold = 0
        self.small_clust_threshold = 0
        self.num_remaining_points = 0
        self.relaxation_factor = hyperparameters[2] #Hyperparameter

    def __compute_costs(self, clustering):
        distances = sklearn.metrics.pairwise_distances(self.cur_points, clustering.cluster_centers_)#, metric="haversine")
        #distances = osrm_dist(self.cur_points, clustering.cluster_centers_, "duration")
        dist_cost = np.zeros(len(clustering.cluster_centers_))
        num_points = np.zeros(len(clustering.cluster_centers_))
        for i in range(len(self.cur_points)):
            dist = distances[i][clustering.labels_[i]]
            #dist_cost[clustering.labels_[i]] += dist
            dist_cost[clustering.labels_[i]] += dist*self.weights[self.cur_point_indices[i]]
            #num_points[clustering.labels_[i]] += 1
            num_points[clustering.labels_[i]] += self.weights[self.cur_point_indices[i]]

        dist_cost = np.divide(dist_cost,num_points)
        #out_file = np.stack((num_points, dist_cost), axis=-1)
        #print(out_file)
        #np.savetxt("C:\\Users\\raghu\\Downloads\\poop" +str(len(clustering.cluster_centers_)) + ".csv", out_file, delimiter = ",")
        dist_cost = np.divide(dist_cost, np.max(dist_cost))
        cop_num_points = np.copy(num_points)
        num_points = np.abs(np.subtract(num_points, self.large_clust_threshold))

        max_cluster_size = np.max(num_points)
        cluster_size = np.divide(num_points, max_cluster_size)

        costs = np.add(np.multiply(dist_cost, self.weight_dist), np.multiply(cluster_size, self.weight_clust_size))

        return((costs,cop_num_points, distances))

    def __accept_clusters(self, clustering, num_to_accept):
        #costs = np.array([self.__compute_costs(clustering, cluster_index) for cluster_index in range(len(clustering.cluster_centers_))])
        (costs,num_points,distances) = self.__compute_costs(clustering)
        #sorted_cluster_indices = np.argpartition(costs, num_to_accept-1)
        sorted_cluster_indices = np.argsort(costs)
        num_points = num_points[sorted_cluster_indices]
        '''pointtt = self.points
        centroidsss = self.cluster_centers_
        labelsss = clustering.labels_'''
        #print(sorted_cluster_indices)
        counter=0
        for i in range(len(sorted_cluster_indices)):
            if num_points[i] >= self.small_clust_threshold:
                ind = sorted_cluster_indices[i]
                self.cluster_centers_[self.num_accepted_clusters] = clustering.cluster_centers_[ind]
                #closest_indices = find_n_closest_points(self.cur_points, clustering.cluster_centers_, clustering.labels_, ind, self.large_clust_threshold)
                closest_indices = find_closest_points(self.cur_points, clustering.cluster_centers_, clustering.labels_, self.cur_weights, distances, ind, self.large_clust_threshold)

                ###print(closest_indices)
                #for points whose label is sorted_cluster_indices[i], set self.labels_ to i and set weights of the points to 0
                #for k in range(len(self.points)):
                for k in closest_indices:
                        k_global = self.cur_point_indices[k]
                    #if k in closest_indices: #if clustering.labels_[k] == sorted_cluster_indices[i]:
                        if(self.labels_[k_global] == -1):
                            self.labels_[k_global] = self.num_accepted_clusters
                            #self.weights[k_global] = 0
                            self.num_remaining_points -= 1
                        else:
                            print("ERROR: re-labeling a point that was already assigned a cluster")
                self.num_accepted_clusters += 1
                counter+=1
            if counter == num_to_accept:
                break

    def fit(self, points, weights):
        self.points = points
        self.num_remaining_points = len(self.points)
        self.weights = weights
        self.labels_ = np.full(len(points), -1, dtype=np.int32)
        self.num_accepted_clusters = 0

        while (self.num_accepted_clusters < self.num_clusters):
            clusters_to_form = self.num_clusters - self.num_accepted_clusters
            clusters_to_accept = int(clusters_to_form*self.fraction_to_accept)
            if(clusters_to_accept < self.min_to_accept):
                clusters_to_accept = clusters_to_form

            total_weight = int(np.sum(self.weights[self.labels_ == -1]))
            print("--- SA-KMeans calling Kmeans to form " + str(clusters_to_form) + " clusters from " + str(self.num_remaining_points) +" points with total weight " + str(total_weight) + " ---")
            clustering = KMeans(n_clusters=clusters_to_form, random_state=self.random_state)
            #clustering.fit_predict(self.points, sample_weight=self.weights)
            #clustering = BisectingKMeans(n_clusters = clusters_to_form, init="k-means++", random_state=self.random_state, bisecting_strategy = "largest_cluster")
            self.cur_point_indices = np.arange(0, len(points), 1, dtype=np.int32)[self.labels_ == -1]
            self.cur_points = self.points[self.labels_ == -1]
            self.cur_weights = self.weights[self.labels_ == -1]
            #print(self.cur_point_indices)
            #print(self.cur_points)
            #print(len(self.cur_points))
            #self.large_clust_threshold = int((len(self.cur_points) *1.25)//clusters_to_form)
            self.large_clust_threshold = int((np.sum(self.cur_weights) * self.relaxation_factor)//clusters_to_form)
            if(clusters_to_form < 100):
                self.small_clust_threshold =0
            else:
                #self.small_clust_threshold = int(np.sum(self.cur_weights)//(clusters_to_form * self.relaxation_factor))
                self.small_clust_threshold = int((np.sum(self.cur_weights)//clusters_to_form) * self.relaxation_factor)
            print("Thresholds: " + str(self.large_clust_threshold) + ", " + str(self.small_clust_threshold))
            clustering.fit_predict(self.cur_points, sample_weight=self.cur_weights)
            self.__accept_clusters(clustering, clusters_to_accept)
            print("--- SA-KMeans accepted total of " + str(self.num_accepted_clusters) + " clusters ---")
            #print(self.cluster_centers_)
        return self

def do_clustering(num_clusters,clustering_method, max_cluster_size, points, weights, hyperparameters):
    # Global parameters
    #num_clusters = 1008
    #"k-means-constrained" #"agglomerative" #"bisecting-kmeans"  #"kmeans" #"dbscan"
    #clustering_method = "kmeans"
    #create_dataset = False
    #points_filename = "points_NY_popdist.csv"
    #weights_filename = "weights_NY_popdist.csv"

    # Create dataset from EV registration and population density data and save it, otherwise just read it
    #if(create_dataset == True):
    #    create_clustering_dataset()
    #else:
    #points = np.loadtxt(points_filename, delimiter=",")
    #weights = np.loadtxt(weights_filename, delimiter=",")

    if(clustering_method == "kmeans"):
        #centroids = KMeans(np.array(points, dtype=float), np.array(rand_centroids, dtype = float))
        clustering = KMeans(n_clusters=num_clusters, random_state=101)
        clustering.fit_predict(points, sample_weight=weights)
    elif(clustering_method == "dbscan"):
        clustering = DBSCAN().fit(points)
    elif(clustering_method == "bisecting-kmeans"):
        clustering = BisectingKMeans(n_clusters = num_clusters, init="k-means++", random_state=101, bisecting_strategy = "largest_cluster")
        clustering.fit(points, sample_weight=weights)
    elif(clustering_method == "agglomerative"):
        clustering = AgglomerativeClustering(n_clusters=num_clusters).fit(points)
    elif(clustering_method == "k-means-constrained"):
        #max_cluster = 500
        #constrained_clusters = int((len(points)/max_cluster)*1.1)
        clustering = KMeansConstrained(n_clusters=num_clusters, size_min=0, size_max=max_cluster_size, random_state=101, n_jobs=-2, max_iter=100, n_init=1, verbose=True)
        clustering.fit_predict(points)
        #print(clustering.cluster_centers_)
    elif(clustering_method == "simple_kmeans"):
        clustering = K_Means(n_clusters=num_clusters, max_iter = 100, random_state = 101)
        clustering.fit(points, sample_weight=weights)
    elif(clustering_method == "sa-kmeans"):
        clustering = SA_KMeans(num_clusters, hyperparameters)#, random_state=101)

        clustering.fit(points, weights)
    else:
        print("ERROR: Unsupported clustering method")
    #print(clustering)
    return(clustering)
    #pickle.dump(clustering, open("models\\model_"+clustering_method+".pickle", 'wb'))
