import numpy as np
x = np.loadtxt("nn_input.csv", delimiter=",")
y = np.loadtxt("nn_output.csv", delimiter=",")


'''y1 = [d[0] for d in y]
y2 = [d[1] for d in y]

y1_min = min(y1)
y2_min = min(y2)

y1_norm = [d-y1_min for d in y1]
y2_norm = [d-y2_min for d in y2]

y1_max = max(y1)
y2_max = max(y2)

y1_norm = [d/y1_max for d in y1]
y2_norm = [d/y2_max for d in y2]

y_norm = list(zip(y1_norm, y2_norm))'''


def filter_coordinates(arr1, arr2):
    # Create a list to store the indices to be kept
    indices_to_keep = []

    # Iterate over each coordinate in the array
    for i in range(len(arr1)):
        keep = True
        for j in range(len(arr1)):
            if i != j and arr1[i][0] >= arr1[j][0] and arr1[i][1] >= arr1[j][1]:
                keep = False
                break
        if keep:
            indices_to_keep.append(i)

    # Keep only the coordinates at the indices to be kept
    arr1 = arr1[indices_to_keep]
    arr2 = arr2[indices_to_keep]

    return arr1, arr2


y_final, x_final = filter_coordinates(y, x)
print(x_final)
print(y_final)
np.savetxt("best_hyperparameters.csv", np.array(x_final), delimiter=",")
np.savetxt("best_hyperparameters_metrics.csv", np.array(y_final), delimiter=",")
