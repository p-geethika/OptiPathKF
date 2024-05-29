#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pykalman import KalmanFilter
import random

fig = plt.figure(figsize=(16, 6))
ax = fig.gca()
plt.ioff()
delete_file = open("kalman.csv","w") # delete the history of random values
delete_file.write("value\n") # create 'value' column in our csv
delete_file.close()

def plot():
    rnd = random.randint(0,1000)
    write_file = open("kalman.csv", "a")
    write_file.write(str(rnd)+"\n")
    write_file.close()
    df = pd.read_csv("kalman.csv")
    x = df.value.tolist()[-100:]# Select only the last 100 values from the dataframe
    cm = x[0:1] 

    cm_seq = np.arange(1,cm[0], step=150)
    cm_list = np.asarray(cm_seq)
    cm_combined = cm_list.tolist()
    cm_combined = cm_combined  + np.asarray(x).tolist()

    kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                      transition_covariance=.1 * np.eye(2))

    states_pred = kf.em(cm_combined).smooth(cm_combined)[0]

    kf2 = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                      transition_covariance=1 * np.eye(2))

    states_pred2 = kf2.em(cm_combined).smooth(cm_combined)[0]


    kf3 = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                      transition_covariance= 2 * np.eye(2))

    states_pred3 = kf3.em(cm_combined).smooth(cm_combined)[0]


    ax.clear()
    ax.plot(cm_combined, ':', label="Random ") 
    ax.plot(states_pred[:, 0], label="Covariance =  0.1")
    ax.plot(states_pred2[:, 0], label="Covariance =  1")
    ax.plot(states_pred3[:, 0], label="Covariance = 2")
    ax.legend(loc=2)


while (True):
    plt.pause(0.001)
    plt.ion()
    plot()

