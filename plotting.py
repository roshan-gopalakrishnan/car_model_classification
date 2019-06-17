""" Developer: Roshan Gopalakrishnan
    Contact: roshan.gopalakrishnan@gmail.com

    Description:
    This code is to plot accuracy vs epochs generated while training.

"""
""" import packages """

import matplotlib.pyplot as plt
import csv
import numpy as np
from itertools import izip

""" Initialize variables """

x = []
y_train_acc = []
y_test_acc = []

""" Open CSV and save the columns needed """

with open('accuracy_vs_epochs.csv','r') as file:
    data = csv.reader(file, delimiter=';')
    for i, row in enumerate(data):
        x.append(i)
        y_train_acc.append(row[1])
        y_test_acc.append(row[3])

""" Plot the result """

p1 = plt.plot(x[1:], y_train_acc[1:], 'k')
p2 = plt.plot(x[1:], y_test_acc[1:], 'b')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.title('Plotting Accuracy')
plt.show()
