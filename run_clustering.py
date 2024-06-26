from cluster_ev_v2 import do_clustering, create_clustering_dataset
from evaluate_clustering import evaluate_clustering
import numpy as np
import pickle
import csv
import os
import json
import time

# Global parameters
cluster_sizes = {"NY": [3771], "MN": [772], "CO": [2133], "CT": [822], "ME": [480], "NJ": [1244], "NC": [1518], "WA": [2135], "VT": [368], "TX": [3159], "OR": [1188], "IN": [542]}
large_cluster_threshold = 75
#"k-means-constrained" #"agglomerative" #"bisecting-kmeans"  #"kmeans" #"dbscan" #"existing" #"simple_kmeans" "sa-kmeans"
clustering_methods = ["sa-kmeans"]
create_dataset = False
#states = ["CO", "CT", "ME", "NJ", "NC", "VT", "TX", "OR"]
states = ["NY"]
#"osrm-distance" "osrm-time" "haversine"


#frac_to_accept = [0.05*(n+1) for n in range(1, 6)]
'''dist_weight = [0.1*(n+1) for n in range(2, 10)]
relaxation_factor = [1+(0.05*n) for n in range(11)]

frac_to_accept = [0.05*(n+1) for n in range(6, 10)]'''

all_hyp = np.loadtxt("best_hyperparameters.csv", delimiter=",")####
all_hyp = list(all_hyp)#####
all_hyp = [0]
frac_to_accept = [0.1*(n+1) for n in range(5)]####
dist_weight = [0.2*(n+1) for n in range(5)]####
relaxation_factor = [1+(0.1*n) for n in range(7)]####

#frac_to_accept += [(0.4*np.random.rand())+0.1 for n in range(175)]
#dist_weight += [(0.9*np.random.rand())+0.1 for n in range(175)]
#relaxation_factor += [(0.5*np.random.rand())+1 for n in range(175)]
#all_hyp = list(np.array(list(zip(frac_to_accept, dist_weight, relaxation_factor))))
#frac_to_accept = [0]
#dist_weight = [0]
#relaxation_factor = [0]

dist_metrics = ["haversine"]

for state in states:
    points_filename = "data\\"+state+"_points_popdist.csv"
    weights_filename = "data\\"+state+"_weights_popdist.csv"
    ev_reg_data_filename = "data\\"+state+"_EV_Registrations.csv"
    zipcode_to_lat_long_filename = "data\\"+"zip_codes_to_lat_long.csv"
    pop_density_filename = "data\\"+state+"_pop_density.csv"
    cur_stations_filename = "data\\"+state+"_current_stations.csv"
    plot_cdf = False

    def create_directory(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    if create_dataset:
        print("------ Creating clustering dataset ------")
        create_clustering_dataset(ev_reg_data_filename, zipcode_to_lat_long_filename, pop_density_filename, points_filename, weights_filename)

    points = np.loadtxt(points_filename, delimiter=",")
    weights = np.loadtxt(weights_filename, delimiter=",")
    x = []
    y = []
    for clustering_method in clustering_methods:
      for hyperparameter_set in all_hyp:######
        for num_clusters in cluster_sizes[state]:
            for frac in frac_to_accept:
                for dist_w in dist_weight:
                    for relax_f in relaxation_factor:
                        hyperparameters = [frac, dist_w, relax_f]
                        #hyperparameters = hyperparameter_set######
                        x.append(hyperparameters)

                        if(clustering_method == "existing"):
                            eval_existing = True
                            print("------ Evaluating existing clusters from " + cur_stations_filename + " ------")
                            folder_name = "models\\"+state+"_2"+clustering_method
                            create_directory(folder_name)
                            model_filename = cur_stations_filename
                            dist_cdf_filename = folder_name+"\\"+state+"_dist_cdf_"+clustering_method+".csv"
                            count_hist_filename = folder_name+"\\"+state+"_count_hist_"+clustering_method+".csv"
                            json_filename = folder_name+"\\"+state+"_centers_"+clustering_method+".json"

                        else:
                            eval_existing = False
                            print("------ Running "+clustering_method + " with "+str(num_clusters)+" clusters on "+ state +" state ------")
                            print("Hyperparamters: ", hyperparameters)
                            folder_name = "models\\"+state+"_"+clustering_method+"_"+str(num_clusters)
                            create_directory(folder_name)
                            model_filename = folder_name+"\\"+state+"_model_"+clustering_method+"_"+str(num_clusters)+".pickle"
                            max_cluster_size = int((len(points)*1.1)/num_clusters)
                            #start_time = time.time()
                            model = do_clustering(num_clusters,clustering_method, max_cluster_size, points, weights, hyperparameters)
                            #print("--- %s seconds ---" % (time.time() - start_time))
                            #continue
                            pickle.dump(model, open(model_filename, 'wb'))
                            dist_cdf_filename = folder_name+"\\"+state+"_dist_cdf_"+clustering_method+"_"+str(num_clusters)+".csv"
                            count_hist_filename = folder_name+"\\"+state+"_count_hist_"+clustering_method+"_"+str(num_clusters)+".csv"
                            json_filename = folder_name+"\\"+state+"_centers_"+clustering_method+"_"+str(num_clusters)+".json"


                        metrics = evaluate_clustering(model_filename, points_filename, weights_filename, plot_cdf=False, evaluate_existing=eval_existing, metrics=dist_metrics)
                        print("Inertia: " + str(metrics[0]))
                        print("Average distance: " + str(metrics[1]))
                        y_mets = metrics[1]


                        count_bin_centers = metrics[4]
                        count_hist = metrics[5]
                        total_large_clusters = sum([count_hist[i] for i in range(len(count_bin_centers)) if int(count_bin_centers[i]) > large_cluster_threshold])

                        print("Fraction large clusters (> " + str(large_cluster_threshold) + "): " + str(total_large_clusters/num_clusters))
                        y_mets.append(total_large_clusters/num_clusters)
                        y.append(y_mets)
                        # Write model to JSON file
                        '''with open(json_filename, 'w') as f:
                            json.dump(metrics[6], f)'''
                        # Save CDF of distances from points to their closest centroid, histogram of cluster sizes
                        '''np.savetxt(dist_cdf_filename, np.transpose(np.array([metrics[2],metrics[3]])), delimiter=",")
                        np.savetxt(count_hist_filename, np.transpose(np.array([metrics[4],metrics[5]])), delimiter=",")'''
            print(x)
            print(y)
            np.savetxt("coarse_hyperparameters.csv", np.array(x), delimiter=",")
            np.savetxt("coarse_clust_metrics.csv", np.array(y), delimiter=",")
