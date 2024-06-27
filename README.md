# EV-Planner
EV-Planner is a tool to place EV charging stations. Here is a one minute elevator pitch: https://youtu.be/Hq33CfAVbVw

The EV-Planner framework consists of three main steps:
- Creating a clustering instance, including pre-processing geospatial data such as EV density
- Performing clustering using the data and verious hyperparameters set by the user
- Evaluating the clustering solution on various metrics and visualizing the solution

![sdfsdf](https://github.com/bharathanand0/EV-Planner/assets/86021254/812244c1-b076-4e76-879b-812a5cec5ea6)

The Python code for all of these steps is included in this repository, and depends on the following libraries/modules: pandas, numpy, pickle, sklearn, geopandas, gzip, shutil k_means_constrained, random, osrm, math, itertools, csv, os, json, and time.

Results can be visualized using an app for mobile devices
This app was written using the Expo Go framework. The app lets you can select a state, clustering algorithm, and the number of stations to place. This will open a map that will show the suggested stations in red and the existing stations in blue. The code for the app is also in the repository (EV_Planner_App.js).

Here is an example the output of EV-Planner for Indiana, with suggested stations in red and existing station in blue (equal number of red and blue):

![image](https://github.com/bharathanand0/EV-Planner/assets/86021254/20d235b4-d8b4-440a-ad74-22866ce4996e)

