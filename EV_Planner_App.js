import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Pressable, TouchableOpacity } from 'react-native';
import React, {useState} from 'react';
import MapView, { Marker } from 'react-native-maps';
import { Image } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import CO_centers_bisecting_kmeans_2133 from "./assets/CO_centers_bisecting-kmeans_2133.json";
import CO_centers_existing from "./assets/CO_centers_existing.json";
import CO_centers_k_means_constrained_2133 from "./assets/CO_centers_k-means-constrained_2133.json";
import CO_centers_kmeans_2133 from "./assets/CO_centers_kmeans_2133.json";
import CO_centers_sa_kmeans_2133 from "./assets/CO_centers_sa-kmeans_2133.json";
import CT_centers_bisecting_kmeans_822 from "./assets/CT_centers_bisecting-kmeans_822.json";
import CT_centers_existing from "./assets/CT_centers_existing.json";
import CT_centers_k_means_constrained_822 from "./assets/CT_centers_k-means-constrained_822.json";
import CT_centers_kmeans_822 from "./assets/CT_centers_kmeans_822.json";
import CT_centers_sa_kmeans_822 from "./assets/CT_centers_sa-kmeans_822.json";
import IN_centers_bisecting_kmeans_542 from "./assets/IN_centers_bisecting-kmeans_542.json";
import IN_centers_existing from "./assets/IN_centers_existing.json";
import IN_centers_k_means_constrained_542 from "./assets/IN_centers_k-means-constrained_542.json";
import IN_centers_kmeans_542 from "./assets/IN_centers_kmeans_542.json";
import IN_centers_sa_kmeans_542 from "./assets/IN_centers_sa-kmeans_542.json";
import ME_centers_bisecting_kmeans_480 from "./assets/ME_centers_bisecting-kmeans_480.json";
import ME_centers_existing from "./assets/ME_centers_existing.json";
import ME_centers_k_means_constrained_480 from "./assets/ME_centers_k-means-constrained_480.json";
import ME_centers_kmeans_480 from "./assets/ME_centers_kmeans_480.json";
import ME_centers_sa_kmeans_480 from "./assets/ME_centers_sa-kmeans_480.json";
import MN_centers_bisecting_kmeans_100 from "./assets/MN_centers_bisecting-kmeans_100.json";
import MN_centers_bisecting_kmeans_500 from "./assets/MN_centers_bisecting-kmeans_500.json";
import MN_centers_bisecting_kmeans_750 from "./assets/MN_centers_bisecting-kmeans_750.json";
import MN_centers_bisecting_kmeans_772 from "./assets/MN_centers_bisecting-kmeans_772.json";
import MN_centers_existing from "./assets/MN_centers_existing.json";
import MN_centers_k_means_constrained_772 from "./assets/MN_centers_k-means-constrained_772.json";
import MN_centers_kmeans_100 from "./assets/MN_centers_kmeans_100.json";
import MN_centers_kmeans_500 from "./assets/MN_centers_kmeans_500.json";
import MN_centers_kmeans_750 from "./assets/MN_centers_kmeans_750.json";
import MN_centers_kmeans_772 from "./assets/MN_centers_kmeans_772.json";
import MN_centers_sa_kmeans_772 from "./assets/MN_centers_sa-kmeans_772.json";
import NC_centers_bisecting_kmeans_1518 from "./assets/NC_centers_bisecting-kmeans_1518.json";
import NC_centers_existing from "./assets/NC_centers_existing.json";
import NC_centers_k_means_constrained_1518 from "./assets/NC_centers_k-means-constrained_1518.json";
import NC_centers_kmeans_1518 from "./assets/NC_centers_kmeans_1518.json";
import NC_centers_sa_kmeans_1518 from "./assets/NC_centers_sa-kmeans_1518.json";
import NJ_centers_bisecting_kmeans_1244 from "./assets/NJ_centers_bisecting-kmeans_1244.json";
import NJ_centers_existing from "./assets/NJ_centers_existing.json";
import NJ_centers_k_means_constrained_1244 from "./assets/NJ_centers_k-means-constrained_1244.json";
import NJ_centers_kmeans_1244 from "./assets/NJ_centers_kmeans_1244.json";
import NJ_centers_sa_kmeans_1244 from "./assets/NJ_centers_sa-kmeans_1244.json";
import NY_centers_bisecting_kmeans_150 from "./assets/NY_centers_bisecting-kmeans_150.json";
import NY_centers_bisecting_kmeans_3771 from "./assets/NY_centers_bisecting-kmeans_3771.json";
import NY_centers_bisecting_kmeans_500 from "./assets/NY_centers_bisecting-kmeans_500.json";
import NY_centers_bisecting_kmeans_750 from "./assets/NY_centers_bisecting-kmeans_750.json";
import NY_centers_existing from "./assets/NY_centers_existing.json";
import NY_centers_kmeans_150 from "./assets/NY_centers_kmeans_150.json";
import NY_centers_kmeans_3771 from "./assets/NY_centers_kmeans_3771.json";
import NY_centers_kmeans_500 from "./assets/NY_centers_kmeans_500.json";
import NY_centers_kmeans_750 from "./assets/NY_centers_kmeans_750.json";
import NY_centers_sa_kmeans_3771 from "./assets/NY_centers_sa-kmeans_3771.json";
import NY_centers_sa_kmeans_500 from "./assets/NY_centers_sa-kmeans_500.json";
import OR_centers_bisecting_kmeans_1188 from "./assets/OR_centers_bisecting-kmeans_1188.json";
import OR_centers_existing from "./assets/OR_centers_existing.json";
import OR_centers_k_means_constrained_1188 from "./assets/OR_centers_k-means-constrained_1188.json";
import OR_centers_kmeans_1188 from "./assets/OR_centers_kmeans_1188.json";
import OR_centers_sa_kmeans_1188 from "./assets/OR_centers_sa-kmeans_1188.json";
import TX_centers_bisecting_kmeans_3159 from "./assets/TX_centers_bisecting-kmeans_3159.json";
import TX_centers_bisecting_kmeans_50 from "./assets/TX_centers_bisecting-kmeans_50.json";
import TX_centers_existing from "./assets/TX_centers_existing.json";
import TX_centers_k_means_constrained_3159 from "./assets/TX_centers_k-means-constrained_3159.json";
import TX_centers_kmeans_3159 from "./assets/TX_centers_kmeans_3159.json";
import TX_centers_sa_kmeans_3159 from "./assets/TX_centers_sa-kmeans_3159.json";
import VT_centers_bisecting_kmeans_368 from "./assets/VT_centers_bisecting-kmeans_368.json";
import VT_centers_bisecting_kmeans_50 from "./assets/VT_centers_bisecting-kmeans_50.json";
import VT_centers_existing from "./assets/VT_centers_existing.json";
import VT_centers_k_means_constrained_368 from "./assets/VT_centers_k-means-constrained_368.json";
import VT_centers_kmeans_368 from "./assets/VT_centers_kmeans_368.json";
import VT_centers_sa_kmeans_368 from "./assets/VT_centers_sa-kmeans_368.json";

const datasets = {
  CO_centers_bisecting_kmeans_2133: CO_centers_bisecting_kmeans_2133,
  CO_centers_existing: CO_centers_existing,
  CO_centers_k_means_constrained_2133: CO_centers_k_means_constrained_2133,
  CO_centers_kmeans_2133: CO_centers_kmeans_2133,
  CO_centers_sa_kmeans_2133: CO_centers_sa_kmeans_2133,
  CT_centers_bisecting_kmeans_822: CT_centers_bisecting_kmeans_822,
  CT_centers_existing: CT_centers_existing,
  CT_centers_k_means_constrained_822: CT_centers_k_means_constrained_822,
  CT_centers_kmeans_822: CT_centers_kmeans_822,
  CT_centers_sa_kmeans_822: CT_centers_sa_kmeans_822,
  IN_centers_bisecting_kmeans_542: IN_centers_bisecting_kmeans_542,
  IN_centers_existing: IN_centers_existing,
  IN_centers_k_means_constrained_542: IN_centers_k_means_constrained_542,
  IN_centers_kmeans_542: IN_centers_kmeans_542,
  IN_centers_sa_kmeans_542: IN_centers_sa_kmeans_542,
  ME_centers_bisecting_kmeans_480: ME_centers_bisecting_kmeans_480,
  ME_centers_existing: ME_centers_existing,
  ME_centers_k_means_constrained_480: ME_centers_k_means_constrained_480,
  ME_centers_kmeans_480: ME_centers_kmeans_480,
  ME_centers_sa_kmeans_480: ME_centers_sa_kmeans_480,
  MN_centers_bisecting_kmeans_100: MN_centers_bisecting_kmeans_100,
  MN_centers_bisecting_kmeans_500: MN_centers_bisecting_kmeans_500,
  MN_centers_bisecting_kmeans_750: MN_centers_bisecting_kmeans_750,
  MN_centers_bisecting_kmeans_772: MN_centers_bisecting_kmeans_772,
  MN_centers_existing: MN_centers_existing,
  MN_centers_k_means_constrained_772: MN_centers_k_means_constrained_772,
  MN_centers_kmeans_100: MN_centers_kmeans_100,
  MN_centers_kmeans_500: MN_centers_kmeans_500,
  MN_centers_kmeans_750: MN_centers_kmeans_750,
  MN_centers_kmeans_772: MN_centers_kmeans_772,
  MN_centers_sa_kmeans_772: MN_centers_sa_kmeans_772,
  NC_centers_bisecting_kmeans_1518: NC_centers_bisecting_kmeans_1518,
  NC_centers_existing: NC_centers_existing,
  NC_centers_k_means_constrained_1518: NC_centers_k_means_constrained_1518,
  NC_centers_kmeans_1518: NC_centers_kmeans_1518,
  NC_centers_sa_kmeans_1518: NC_centers_sa_kmeans_1518,
  NJ_centers_bisecting_kmeans_1244: NJ_centers_bisecting_kmeans_1244,
  NJ_centers_existing: NJ_centers_existing,
  NJ_centers_k_means_constrained_1244: NJ_centers_k_means_constrained_1244,
  NJ_centers_kmeans_1244: NJ_centers_kmeans_1244,
  NJ_centers_sa_kmeans_1244: NJ_centers_sa_kmeans_1244,
  NY_centers_bisecting_kmeans_150: NY_centers_bisecting_kmeans_150,
  NY_centers_bisecting_kmeans_3771: NY_centers_bisecting_kmeans_3771,
  NY_centers_bisecting_kmeans_500: NY_centers_bisecting_kmeans_500,
  NY_centers_bisecting_kmeans_750: NY_centers_bisecting_kmeans_750,
  NY_centers_existing: NY_centers_existing,
  NY_centers_kmeans_150: NY_centers_kmeans_150,
  NY_centers_kmeans_3771: NY_centers_kmeans_3771,
  NY_centers_kmeans_500: NY_centers_kmeans_500,
  NY_centers_kmeans_750: NY_centers_kmeans_750,
  NY_centers_sa_kmeans_3771: NY_centers_sa_kmeans_3771,
  NY_centers_sa_kmeans_500: NY_centers_sa_kmeans_500,
  OR_centers_bisecting_kmeans_1188: OR_centers_bisecting_kmeans_1188,
  OR_centers_existing: OR_centers_existing,
  OR_centers_k_means_constrained_1188: OR_centers_k_means_constrained_1188,
  OR_centers_kmeans_1188: OR_centers_kmeans_1188,
  OR_centers_sa_kmeans_1188: OR_centers_sa_kmeans_1188,
  TX_centers_bisecting_kmeans_3159: TX_centers_bisecting_kmeans_3159,
  TX_centers_bisecting_kmeans_50: TX_centers_bisecting_kmeans_50,
  TX_centers_existing: TX_centers_existing,
  TX_centers_k_means_constrained_3159: TX_centers_k_means_constrained_3159,
  TX_centers_kmeans_3159: TX_centers_kmeans_3159,
  TX_centers_sa_kmeans_3159: TX_centers_sa_kmeans_3159,
  VT_centers_bisecting_kmeans_368: VT_centers_bisecting_kmeans_368,
  VT_centers_bisecting_kmeans_50: VT_centers_bisecting_kmeans_50,
  VT_centers_existing: VT_centers_existing,
  VT_centers_k_means_constrained_368: VT_centers_k_means_constrained_368,
  VT_centers_kmeans_368: VT_centers_kmeans_368,
  VT_centers_sa_kmeans_368: VT_centers_sa_kmeans_368,
  
}



import { RFValue } from 'react-native-responsive-fontsize';
import {Picker} from '@react-native-picker/picker';
import Checkbox from 'expo-checkbox';
import { HeaderBackButton } from '@react-navigation/elements';
import { useNavigation } from '@react-navigation/native';

function StartScreen({ navigation }) {
  return (
    <View style={styles.startcontainer}>
      <Text style = {styles.bigboldtext}>EVPlanner</Text>
      <Text></Text>
      <Text></Text>
      <Text></Text>
      <Text></Text>
      <TouchableOpacity onPress={() => navigation.navigate("Select Options")}>
        <Image style = {styles.logo} source = {require('./assets/Picture1_trnsp_alt.png')}/>
      </TouchableOpacity>
    </View>
  );
}

function MenuScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <Text style = {styles.bigboldtext}>EVPlanner</Text>

      <Image style = {styles.logo} source = {require('./assets/Picture1.jpg')}/>
      <Text></Text>
      <Pressable
        style={({pressed}) => [
          {
            opacity: pressed ? 0.5 : 1
          },
          styles.button,
        ]}
        onPress={() => navigation.navigate("Resources")}>
        <Text style={styles.buttonText}>Resources</Text>
      </Pressable>
      <Text></Text>
      <Pressable
        style={({pressed}) => [
          {
            opacity: pressed ? 0.5 : 1
          },
          styles.button,
        ]}
        onPress={() => navigation.navigate("Daily Screening")}>
        <Text style={styles.buttonText}>Daily Screening</Text>
      </Pressable>
      <Text></Text>
      <Pressable
        style={({pressed}) => [
          {
            opacity: pressed ? 0.5 : 1
          },
          styles.button,
        ]}
        onPress={() => navigation.navigate("Past Results")}>
        <Text style={styles.buttonText}>Screening History</Text>
      </Pressable>
      <Text></Text>
      <Pressable
        style={({pressed}) => [
          {
            opacity: pressed ? 0.5 : 1
          },
          styles.button,
        ]}
        onPress={() => navigation.navigate("Find Treatment")}>
        <Text style={styles.buttonText}>Find Treatment</Text>
      </Pressable>
      <Text></Text>
      <Pressable
        style={({pressed}) => [
          {
            opacity: pressed ? 0.5 : 1
          },
          styles.button,
        ]}
        onPress={() => navigation.navigate("About Us")}>
        <Text style={styles.buttonText}>About Us</Text>
      </Pressable>
      <Text style = {styles.redTitle}>If you are having a mental health crisis, immediately call 988!</Text>
    </View>
  );
}

function Resources() {
  const [selectedValue1, setSelectedValue1] = useState('value1');
  const [selectedValue2, setSelectedValue2] = useState('value2');
  const [selectedValue3, setSelectedValue3] = useState('value3');
  const [num_clust_filtered, setNumClustFiltered] = useState(["Select no. of stations"]);
  const [isSelected, setSelection] = useState(false);
  const navigation = useNavigation();
  const states = ["Select a State", "Colorado", "Connecticut", "Indiana", "Maine", "Minnesota", "North Carolina", "New Jersey", "New York", "Oregon", "Texas", "Vermont"];
  const state_values = ["Select a State", "CO", "CT", "IN", "ME", "MN", "NC", "NJ", "NY", "OR", "TX", "VT"];

  const algorithms = ["Select an Algorithm", "KMeans", "Bisecting KMeans", "Constrained KMeans", "SA Clustering"];
  const alg_values = ["Select an Algorithm", "kmeans", "bisecting_kmeans", "k_means_constrained", "sa_kmeans"]
  var num_clusters = [["Select no. of stations"], ["Select no. of stations", "2133"], ["Select no. of stations", "822"], ["Select no. of stations", "542"], ["Select no. of stations", "480"], ["Select no. of stations", "772"], ["Select no. of stations", "1518"], ["Select no. of stations", "1244"], ["Select no. of stations", "3771"], ["Select no. of stations", "1188"], ["Select no. of stations", "3159"], ["Select no. of stations", "368"]];
  var num_clust_values = num_clusters
  const handleFirstPickerChange = (value) => {
    setSelectedValue1(value);
    const index = state_values.indexOf(value);
    setNumClustFiltered(num_clusters[index]);
  };
  return (
    <View style={styles.dropdowncontainer}>
      <View style={{ borderWidth: 3, borderColor: '#000', borderRadius: 10 }}>
        {/* Picker 1 */}
        <Picker
          selectedValue={selectedValue1}
          style={{height: 40, width: 250}}
          onValueChange={handleFirstPickerChange}
        >
          {states.map((label, index) => (
            <Picker.Item key={index} label={label} value={state_values[index]} />
          ))}
        </Picker>
      </View>
      <Text></Text>
      <View style={{ borderWidth: 3, borderColor: '#000', borderRadius: 10 }}>
        {/* Picker 2 */}
        
        <Picker
          selectedValue={selectedValue2}
          style={{height: 40, width: 250}}
          onValueChange={(itemValue) => setSelectedValue2(itemValue)}
        >
          {algorithms.map((label, index) => (
            <Picker.Item key={index} label={label} value={alg_values[index]} />
          ))}
        </Picker>
      </View>
      <Text></Text>
      <View style={{ borderWidth: 3, borderColor: '#000', borderRadius: 10 }}>
        {/* Picker 3 */}
        <Picker
          selectedValue={selectedValue3}
          style={{height: 40, width: 250}}
          onValueChange={(itemValue) => setSelectedValue3(itemValue)}
        >
  {num_clust_filtered.map((label, index) => (
    <Picker.Item key={index} label={label.toString()} value={label.toString()} />
  ))}
</Picker>
      </View>
      <Text></Text>
      {/* Checkbox */}
      <View style={{ flexDirection: 'row', alignItems: 'center' }}>
        <Checkbox
          value={isSelected}
          onValueChange={setSelection}
        />
        <Text style= {{fontSize: RFValue(14)}}>{'  Plot Existing Stations'}</Text>
      </View>
      <Text></Text>
      <Text></Text>
      {/* Navigation Button */}
      <Pressable
        style={({pressed}) => [
          {
            opacity: pressed ? 0.5 : 1
          },
          styles.button,
        ]}
        onPress={() => {
          const routeName = isSelected ? 'with_existing' : 'without_existing';
          const params = {
            result: `${selectedValue1}_centers_${selectedValue2}_${selectedValue3}`,
          };
          navigation.navigate(routeName, params);
        }}>
        <Text style={styles.buttonText}>Map Stations</Text>
      </Pressable>
    </View>
      

  );
}


function With_existing({route}) {
  const {result} = route.params;
  
  const [tracksViewChanges, setTracksViewChanges] = useState(true);
  const markers1 = datasets[result].map((pair, index) => ({
    id: `1-${index}`, // FlatList requires a unique key
    latitude: pair[0],
    longitude: pair[1],
  }));
  
  const markers2 = datasets[result.substring(0,2)+"_centers_existing"].map((pair, index) => ({
    id: `2-${index}`, // FlatList requires a unique key
    latitude: pair[0],
    longitude: pair[1],
  }));
  const handleLoad = () => {
    setTracksViewChanges(false);
  };
  

  return (
    <MapView
      style={{ flex: 1 }}
      initialRegion={{
        latitude: 42.6526,
        longitude: -73.7562,
        latitudeDelta: 10,
        longitudeDelta: 10,
      }}
    >
      {markers1.map((marker) => (
        <Marker
          key={marker.id}
          coordinate={{ latitude: marker.latitude, longitude: marker.longitude }}
          anchor={{ x: 0.5, y: 0.5 }}
          tracksViewChanges={tracksViewChanges}
        >
          <Image
            source={require('./assets/red_marker.png')}
            style={{
              width: 7.5,
              height: 7.5,
            }}
            resizeMode="cover"
            onLoadEnd={handleLoad}
            fadeDuration={0}
          />
        </Marker>
      ))}
      {markers2.map((marker) => (
        <Marker
          key={marker.id}
          coordinate={{ latitude: marker.latitude, longitude: marker.longitude }}
          anchor={{ x: 0.5, y: 0.5 }}
          tracksViewChanges={tracksViewChanges}
        >
          <Image
            source={require('./assets/green_marker.png')}
            style={{
              width: 7.5,
              height: 7.5,
            }}
            resizeMode="cover"
            onLoadEnd={handleLoad}
            fadeDuration={0}
          />
        </Marker>
      ))}
    </MapView>
  );
}
function Without_existing({route}) {
  const {result} = route.params;
  
  const [tracksViewChanges, setTracksViewChanges] = useState(true);
  const markers1 = datasets[result].map((pair, index) => ({
    id: `1-${index}`, // FlatList requires a unique key
    latitude: pair[0],
    longitude: pair[1],
  }));
  

  const handleLoad = () => {
    setTracksViewChanges(false);
  };
  

  return (
    <MapView
      style={{ flex: 1 }}
      initialRegion={{
        latitude: 42.6526,
        longitude: -73.7562,
        latitudeDelta: 10,
        longitudeDelta: 10,
      }}
    >
      {markers1.map((marker) => (
        <Marker
          key={marker.id}
          coordinate={{ latitude: marker.latitude, longitude: marker.longitude }}
          anchor={{ x: 0.5, y: 0.5 }}
          tracksViewChanges={tracksViewChanges}
        >
          <Image
            source={require('./assets/red_marker.png')}
            style={{
              width: 7.5,
              height: 7.5,

            }}
            resizeMode="cover"
            onLoadEnd={handleLoad}
            fadeDuration={0}
          />
        </Marker>
      ))}
    </MapView>
  );
}





const Stack = createNativeStackNavigator();
function App() {
  
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Start" component={StartScreen} options={{
            headerShown: false
          }}/>
        <Stack.Screen name="Menu" component={MenuScreen}/>
        <Stack.Screen name="Select Options" component={Resources} options = {{headerTitleAlign: 'center', headerStyle: { backgroundColor: 'green' }}}/>
        
        <Stack.Screen
          name="with_existing"
          component={With_existing}
          options={({ navigation }) => ({
            title: "Return to Menu",
            headerLeft: (props) => (
              <HeaderBackButton
                {...props}
                onPress={() => {
                  // Navigate to a custom route
                  navigation.navigate("Start");
                }}
              />
            ),
            headerStyle: { backgroundColor: 'green' }
          })}
        />
        <Stack.Screen
          name="without_existing"
          component={Without_existing}
          options={({ navigation }) => ({
            title: "Return to Menu",
            headerLeft: (props) => (
              <HeaderBackButton
                {...props}
                onPress={() => {
                  // Navigate to a custom route
                  navigation.navigate("Start");
                }}
              />
            ),
            headerStyle: { backgroundColor: 'green' }
          })}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    marginBottom: 5
  },
  startcontainer: {
    flex: 1,
    backgroundColor: '#c8c8c8',
    justifyContent: "center",
    alignItems: 'center',
    marginBottom: 5
  },
  dropdowncontainer: {
    flex: 1,
    backgroundColor: '#c8c8c8',
    justifyContent: 'center',
    flexDirection: 'column',
    alignItems: "center",
  },
  bodytext: {
    fontSize: RFValue(16),
    justifyContent: 'flex-start',
    alignItems: 'flex-start'
  },
  question: {
    flex: 1,
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginVertical: 15,
  },
  questionText: {
    fontSize: RFValue(19),
    color: 'black',
    textAlign: 'left',
    margin: 20,
  },
  answerChoiceBack: {
    flex: 0.6,
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginVertical: 15
  },
  answerChoice: {
    backgroundColor: "white",
    borderColor: "black",
    borderWidth: 2,
    borderRadius: 0,
    padding: 10,
    height: "auto",
    width: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 5,
    marginBottom: 10,  // Add space below each Pressable
  },
  answerChoiceText: {
    fontSize: RFValue(16),
    color: 'black',
  },
  bigboldtext: {
    fontSize: RFValue(40),
    fontWeight: "bold",
    color: "#39c213"
  },
  title: {
    textAlign: 'center',
    marginVertical: 8,
    fontSize: RFValue(25),
  },
  redTitle: {
    textAlign: 'center',
    marginVertical: 8,
    fontSize: RFValue(20),
    color: "red",
    marginLeft: 10,
    marginRight: 10,
    marginTop: 20,
  },
  logo: {
    width: "20%",
    padding: 0,
    alignItems: 'center',
    justifyContent: 'center',
    aspectRatio: 0.95,
    maxWidth: "72%",
    maxHeight: "62%"
  },
  fixToText: {
    flex: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    backgroundColor: "white",
    width: "90%"
  },
  button: {
    backgroundColor: "green",
    borderRadius: 8,
    padding: 10,
    height:"auto",
    width: 250,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 5,
  },
  resourcebutton: {
    backgroundColor: "#A4C2F4",
    borderRadius: 8,
    padding: 6,
    margin: 12,
    height:"auto" ,
    width: '90%',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 5,
  },
  previousnext: {
    backgroundColor: "#A4C2F4",
    borderRadius: 8,
    padding: 6,
    height:"auto" ,
    width: '40%',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 5,
  },
  backtoq9: {
    backgroundColor: "#A4C2F4",
    borderRadius: 8,
    padding: 6,
    height:"auto" ,
    width: '70%',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 5,
  },
  buttonText: {
    fontSize: RFValue(20),
    color: 'black',
  },
  buttonTextSmall: {
    fontSize: RFValue(14),
    color: '#616161',
  },
  separator: {
    marginVertical: 8,
    borderBottomColor: '#737373',
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  textInput: {
    margin: 15,
    paddingTop: 2,
    paddingBottom: 2,
    paddingLeft: 7,
    paddingRight: 7,
    borderColor: "black",
    borderWidth: 2,
  },
});


export default App;
