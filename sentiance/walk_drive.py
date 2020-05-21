"""
https://physics.stackexchange.com/questions/316178/how-to-remove-gravity-component-from-accelerometer-x-y-readings

label 1 is walking
label 0 is driving

Walking is very regular, mostly you have your cell phone in your pocket, so the gravitational force is most likely not on the z-axis
For label 0 the z-axis mostly shows the gravitational acceleration, maybe in most cars the cell phone lies on its back
If you are on the highway it can be that there is very little acceleration

Sample frequency is 26Hz, can be determined by:
  - number of samples divided by 60 (as the data contains 60 seconds of data)
  - based on the the time difference between the samples (assuming time is given in nanoseconds)

Seems like the cut-off frequency for low-pass filter is around 7Hz
Walking seems to have high power at 0.75Hz and a bit below 2Hz
There are 780 points in the fft which corespond to 0-13Hz
So 0.75Hz is around sample 45 of the fft and 2Hz around sample 120
"""
import pandas as pd
import numpy as np

from scipy.fftpack import fft 

from sklearn.svm import LinearSVC

NANO = 10**9
TIME_COLUMN = "seg_1_taxis"
COLUMN_TEMPLATES = ["seg_{seg_id}_x", "seg_{seg_id}_y", "seg_{seg_id}_z", "seg_{seg_id}_taxis"]

def load_data(sensor_file, label_file):
  sensors_df = pd.read_csv(sensor_file)
  labels_df = pd.read_csv(label_file)

  return sensors_df, labels_df

def determine_sample_frequency(df, time_column=TIME_COLUMN, time_resolution=NANO):
  time = df[time_column].values
  time_step = np.unique(np.diff(time))[0]
  sample_frequency = np.round(time_resolution/time_step)

  return sample_frequency

def filter_gravity(signal):
  """
  Filter out the gravity offset component

  https://developer.android.com/reference/android/hardware/SensorEvent.html
  """
  signal_mean = np.mean(signal)
  return signal-signal_mean

def preprocess_data(sensors_df, labels_df, n_segments, n_samples):
  """
  :returns: total accelerations (size of vector) and array of labels
  """
  total_accelerations = np.zeros((n_segments, n_samples))
  labels = np.zeros((n_segments,))

  for seg_id in range(1,n_segments+1):
    columns = [name.format(seg_id=seg_id) for name in COLUMN_TEMPLATES[0:3]]
    
    accelerations = sensors_df[columns].values
    label = labels_df[labels_df["segment_id"]==f"seg_{seg_id}"]["label"].values[0]
    
    for i in range(3):
      accelerations[:,i] = filter_gravity(accelerations[:,i])
    
    total_accelerations[seg_id-1, :] = np.sqrt(np.sum(np.power(accelerations,2),axis=1))
    labels[seg_id-1] = label

  return total_accelerations, labels

def compute_fft(signal, sample_frequency):
    N = len(signal)

    xF = np.linspace(0.0, sample_frequency/2.0, int(N/2))
    signalF = fft(signal) 

    return xF, (2/N)*np.abs(signalF[:int(N/2)])

def compute_ffts(total_accelerations, sample_frequency):
  n_segments = total_accelerations.shape[0]
  n_samples = total_accelerations.shape[1]

  ffts = np.zeros((n_segments, int(n_samples/2)))
  for seg_id in range(0,n_segments):
      xF, ffts[seg_id,:] = compute_fft(total_accelerations[seg_id,:], sample_frequency)

  return xF, ffts

def largest_monotonic_sequence(signal, sign=1):
  length_largest_sequence = 0
  start_largest_sequence = 0
  
  sequence_length = 0
  
  prev_diff = -20
  for i, val in enumerate(signal[1:]):
    diff = val-signal[i-1]
        
    if sign*prev_diff >= 0 and sign*diff < 0: #a top if sign == 1
      if sequence_length>length_largest_sequence:
        length_largest_sequence = sequence_length
        start_largest_sequence = i-sequence_length
      sequence_length = 0
        
    if sign*diff>=0:
      sequence_length += 1
        
    prev_diff = diff
  
  return start_largest_sequence, length_largest_sequence

def compute_feature_matrix(total_accelerations, sample_frequency):
  n_segments = total_accelerations.shape[0]
  n_samples = total_accelerations.shape[1]
  
  feature_matrix = np.zeros((n_segments, 2))
  
  lb = int(0.75*(n_samples/sample_frequency))
  ub = int(2*(n_samples/sample_frequency))

  xF, ffts = compute_ffts(total_accelerations, sample_frequency)
  feature_matrix[:,0] = ffts[:,lb:ub].sum(axis=1)
  
  for i in range(n_segments):
      start_index, length = largest_monotonic_sequence(total_accelerations[i,:])
      start_index_decrease, length_decrease = largest_monotonic_sequence(total_accelerations[i,:],-1)
      feature_matrix[i,1] = max(length,length_decrease)/sample_frequency

  return feature_matrix

def build_classifier(feature_matrix, labels):
  classifier = LinearSVC()
  classifier.fit(feature_matrix, labels)
  return classifier

def compute_decision_boundary(classifier, feature_matrix):
  x = np.linspace(0,12,1000)
  y = (1/classifier.coef_[0,1])*(-classifier.coef_[0,0]*x - classifier.intercept_)
  
  lby = np.min(feature_matrix[:,1])
  uby = np.max(feature_matrix[:,1])
  
  x = x[(y>=lby) & (y<=uby)]
  y = y[(y>=lby) & (y<=uby)]

  return x, y

if __name__ == "__main__":
  sensors_df, labels_df = load_data(
      "../data/sensors.csv",
      "../data/labels.csv"
  )
  
  n_segments = int(sensors_df.shape[1]/4)
  n_samples = sensors_df.shape[0]

  sample_frequency = determine_sample_frequency(sensors_df)
  total_accelerations, labels = preprocess_data(
      sensors_df, 
      labels_df, 
      n_segments, 
      n_samples
  )

  feature_matrix = compute_feature_matrix(total_accelerations, sample_frequency)

  model = build_classifier(feature_matrix, labels)

