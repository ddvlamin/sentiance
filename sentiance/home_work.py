import pickle
import json
from collections import Counter

import pandas as pd
import numpy as np

from sklearn.cluster import MeanShift

from sentiance.location import Location, decode_geohash

NBITS = 30
MIN_DAY_DENSITY = 0.5
MIN_DURATION = 1
MAX_DIST = 200

def load_person_data(filepath, sep=";"):
  return pd.read_csv(filepath,sep=sep).rename(
      columns={
        "start_time(YYYYMMddHHmmZ)":"start_time",
        "duration(ms)": "duration"
      }
    )

def add_columns(df, df_format='%Y%m%d%H%M%z', geohash_precision=NBITS):
  df['start_time'] = pd.to_datetime(df['start_time'], format=df_format)
  df["date"] = df['start_time'].apply(lambda x: x.date())
  df["dayofweek"] = df['start_time'].apply(lambda x: x.dayofweek)
  df["hourofday"] = df['start_time'].apply(lambda x: x.hour)
  df["hour_duration"] = df['duration'].apply(lambda x: x/(3600000))

  def _compute_geohash(row, nbits=geohash_precision):
    loc = Location(row["latitude"], row["longitude"], nbits)
    return loc.geohash
    
  df["geohash"] = df.apply(_compute_geohash, axis=1)

  return df

def weekday_counts(df):
  weekday_counter = Counter()

  unique_days = df["date"].unique()
  for dt in unique_days:
    weekday_counter[dt.weekday()] += 1

  return weekday_counter

def create_location_featureset(df, weekday_counter, min_day_density=MIN_DAY_DENSITY):
  """
  :returns DataFrame: each record represents a location with following columns:
    geohash: the geohash computed based on the give lat and long
    day_frequency: percentage of days that this geo location pops up
    mean_duration: the average duration one stays at that location
    mean_starthour: the average hour of the day one arrives at that location
    mean_lat, mean_lng: the average of latitude and longitude measurements of this location (should reduce some measurement variance)
    number_of_days: the number of weekdays that the persons visits this location with a minimum percentage of *min_day_density* 
  """
  n_unique_days = sum(weekday_counter.values())

  hash_df = []
  for geohash, group in df.groupby("geohash"):
    record = {
      "geohash": geohash,
      "day_frequency": len(group["date"].unique())/n_unique_days,
      "mean_duration": group["hour_duration"].mean(),
      "mean_starthour": group["hourofday"].mean(),
      "mean_lat": group["latitude"].mean(),
      "mean_lng": group["longitude"].mean()
    }
    
    #counts for each day in the week how many times it occurs uniquely across the period
    week_hist = Counter()
    for date, g in group.groupby("date"):
      weekday = date.weekday()
      week_hist[weekday] += 1/weekday_counter[weekday]
          
    record["number_of_days"] = len([day for day, density in week_hist.items() if density>min_day_density])
    
    hash_df.append(record)

  return pd.DataFrame(hash_df)

def filter_records(df, min_duration=MIN_DURATION):
  """
  Filter out locations where the person has been (on average) no longer than min_duration
  and where the visit frequency of that location is below the mean of all locations
  """
  mean_frequency = np.mean(df["day_frequency"].unique())
  filtered_df = df[(df["mean_duration"]>=min_duration) &
                   (df["day_frequency"]>mean_frequency)]

  return filtered_df

def compute_distances(df):
  distances = []
  for i in range(df.shape[0]):
    rec1 = df.iloc[i,:].to_dict()
    loc1 = Location(rec1["mean_lat"], rec1["mean_lng"], NBITS)
    for j in range(i+1,df.shape[0]):
      rec2 = df.iloc[j,:].to_dict()
      loc2 = Location(rec2["mean_lat"], rec2["mean_lng"], NBITS)
      dist = loc1.distance(loc2)
      distances.append({
          "src": rec1["geohash"],
          "dst": rec2["geohash"],
          "dist": dist
      })

  return pd.DataFrame(distances)

def mean_location_distance(df):
  """
  Computes for each location the mean distance to the other locations

  :returns DataFrame: index is the geohash with column dist for the average distance to the other locations
  """
  distances = compute_distances(df)

  dist1 = distances[["dst","dist"]].rename(columns={"dst":"src"})
  dist2 = distances[["src","dist"]]
  distances = pd.concat((dist1,dist2), axis=0)

  return distances.groupby("src").agg(np.mean)

def remove_outlier_locations(df, max_dist=MAX_DIST):
  """
  Removes locations that are very far away (defined by *max_dist*) from the other locations
  
  :param max_dist: maximum mean distance to other locations in km
  """
  distances = mean_location_distance(df)
  geohash_toremove = set(distances[distances["dist"]>=max_dist].index)
  df=df[~df["geohash"].isin(geohash_toremove)]

  return df

def create_feature_matrix(
    filepath,
    geohash_precision=NBITS, 
    min_day_density=MIN_DAY_DENSITY):

  df = load_person_data(filepath)
  df = add_columns(df,geohash_precision=geohash_precision)
  weekday_counter = weekday_counts(df)
  df = create_location_featureset(df, weekday_counter, min_day_density=min_day_density)

  return df

def create_output(home_df, work_df):
  result = {
    "home": [],
    "work": []
  }
  for _, row in home_df.iterrows():
    result["home"].append((row["mean_lat"],row["mean_lng"]))

  for _, row in work_df.iterrows():
    result["work"].append((row["mean_lat"],row["mean_lng"]))

  return result

def probable_home_work_locations(
      filepath, 
      geohash_precision=NBITS, 
      min_day_density=MIN_DAY_DENSITY, 
      min_duration=MIN_DURATION,
      max_dist=MAX_DIST):
 
  df = create_feature_matrix(filepath, geohash_precision, min_day_density)
  df = filter_records(df, min_duration=min_duration)
  df = remove_outlier_locations(df, max_dist=max_dist)

  home_df = df[df["number_of_days"]>=6][["mean_lat","mean_lng"]]
  work_df = df[df["number_of_days"]<6][["mean_lat","mean_lng"]]

  return create_output(home_df, work_df)

def build_cluster_model(
      filepath, 
      geohash_precision=NBITS, 
      min_day_density=MIN_DAY_DENSITY):
  
  df = create_feature_matrix(filepath, geohash_precision, min_day_density)
  
  feature_matrix = df[["day_frequency","mean_duration","number_of_days"]].values
  feature_matrix[:,1] /= 24
  feature_matrix[:,2] /= 7
  
  clusterer = MeanShift()
  clusterer.fit(feature_matrix)

  return clusterer

def apply_cluster_model( 
      filepath, 
      modelfile,
      geohash_precision=NBITS, 
      min_day_density=MIN_DAY_DENSITY):
  
  df = create_feature_matrix(filepath, geohash_precision, min_day_density)

  with open(modelfile,"rb") as fin:
    clusterer = pickle.load(fin)
  
  feature_matrix = df[["day_frequency","mean_duration","number_of_days"]].values
  feature_matrix[:,1] /= 24
  feature_matrix[:,2] /= 7
  
  clusters = clusterer.predict(feature_matrix)

  home_df = df.iloc[clusters==4,:] #home
  work_df = pd.concat([df.iloc[clusters==5,:],df.iloc[clusters==6,:]])

  return create_output(home_df, work_df)

if __name__ == "__main__":
  home_work_1 = probable_home_work_locations("../data/Copy of person.1.csv")
  print("Probably home an work locations for person 1")
  print(json.dumps(home_work_1, indent=2))
  home_work_2 = probable_home_work_locations("../data/Copy of person.2.csv")
  print("Probably home an work locations for person 2")
  print(json.dumps(home_work_2, indent=2))
  home_work_3 = probable_home_work_locations("../data/Copy of person.3.csv")
  print("Probably home an work locations for person 3")
  print(json.dumps(home_work_3, indent=2))
