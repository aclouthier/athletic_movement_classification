# athletic_movement_classification

Sensor data required for automatic recognition of athletic tasks using deep neural networks.
Authors: AL Clouthier, GB Ross, RB Graham

athletic_movement_classification.py: This code automatically classifies the movement being performed in each frame 
of data using DNNs previously trained on subsets of the sIMU data.

movement_screen_data.pickle contains the simulated IMU data all 13 body segments 
for three athletes performing 13 athletic tasks. 

Tested using:
    Python 3.7.3
    pytorch 1.1.0
    numpy 1.16.2
    scikit-learn 0.20.3
    scipy 1.2.1 
    matplotlib 3.0.3
