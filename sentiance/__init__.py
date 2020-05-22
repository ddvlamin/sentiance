import os

thelibFolder = os.path.dirname(os.path.realpath(__file__))

work_home_model_file = os.path.join(thelibFolder,"..","models","work_home_classifier.bin")
person_file1 = os.path.join(thelibFolder,"..","data","Copy of person.1.csv")
person_file2 = os.path.join(thelibFolder,"..","data","Copy of person.2.csv")
person_file3 = os.path.join(thelibFolder,"..","data","Copy of person.3.csv")
sensor_file = os.path.join(thelibFolder,"..","data","sensors.csv")
labels_file = os.path.join(thelibFolder,"..","data","labels.csv")


