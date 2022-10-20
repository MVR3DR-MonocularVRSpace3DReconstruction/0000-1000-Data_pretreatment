
from cProfile import label
import time
import numpy as np
import csv,datetime
import matplotlib.pyplot as plt
from datetime import datetime


#####################################
# Load Data Function
#####################################

class CameraPose:
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat
    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)
def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        data = f.read().split('\n')
        for idx in range(0,len(data)//6*6,6):
            meta = data[idx]
            mat = [ [ float(num) for num in row.strip(' ').split(' ')  ] for row in data[idx+1:idx+5] ]
            traj.append(CameraPose(meta, mat))
    return traj


if __name__ == '__main__':
    ###################################
    # Prepare Data
    ###################################

    gyro_labels = ['w_x','w_y','w_z']
    acce_labels = ['a_x','a_y','a_z']
    magnet_labels = ['T_x','T_y','T_z']
    imu = read_trajectory("./imu_data/Rotate.txt")

    ###################################
    # Record New Data 
    ###################################

    data_time = np.array([ datetime.strptime(ii.metadata, '%Y-%m-%d %H:%M:%S.%f')  for ii in imu])
    data_time = np.array([(ii - data_time[0]).total_seconds() for ii in data_time])
    # print(data_time)
    average_delta_time = data_time[-1] / len(data_time)
    print("average delta time: {}// {}Hz".format(average_delta_time,1/average_delta_time))
    data_time_deviation = [ii*average_delta_time - data_time[ii] for ii in range(len(data_time))]
    data_time_deviation_abs = [abs(ii) for ii in data_time_deviation]
    print("Max deviation:{}\tMin deviation:{}".format(max(data_time_deviation_abs), min(data_time_deviation_abs)))
    # print(data_time_deviation_abs)

    data_acce_g = np.array([ii.pose[0] for ii in imu])
    data_acce_u = np.array([ii.pose[1] for ii in imu])
    data_gyro = np.array([ii.pose[2] for ii in imu])
    data_magnet = np.array([ii.pose[3] for ii in imu])

    time_delta = [ data_time[idx+1] - data_time[idx]  for idx in range(len(data_time)-1)]
    time_delta.insert(0, 0)


    

    title_list = [['accelero meter Values',''],
                  ['user Accelero meter Values',''],
                  ['gyroscope Values','Integtrate gyroscope Values'],
                  ['magneto meter Values','']]

    label_list = [['$a_{x,y,z}$ [$m^{2}/s$]',''],
                  ['$a_{x,y,z}$ [$m^{2}/s$]',''],
                  ['$w_{x,y,z}$ [$^{\circ}/s$]','$w_{x,y,z}$ [$^{\circ}/s$]'],
                  ['$T_{x,y,z}$','']]

    ###################################
    # Integrate Data
    ###################################

    data_gyro_inte = []
    angleX = 0; angleY = 0; angleZ = 0
    # print(time_delta)
    for gyro in data_gyro:
        dt = 1
        angleX += gyro[0]*dt; angleY += gyro[1]*dt; angleZ += gyro[2]*dt
        data_gyro_inte.append([angleX, angleY, angleZ])
    data_gyro_inte = np.array(data_gyro_inte)
    # print(data_gyro_inte)

    



    ###################################
    # Plot with and without offsets
    ###################################
    
    plt.style.use('ggplot')
    # x row, y column// x width, y height
    # axs[n row][m col]
    fig, axs = plt.subplots(4,2,figsize=(24,18))

    for ii in range(0,3):
        axs[0][0].plot(data_acce_g[:,ii], label='${}$'.format(acce_labels[ii]))
        axs[1][0].plot(data_acce_u[:,ii], label='${}$'.format(acce_labels[ii]))
        axs[2][0].plot(data_gyro[:,ii], label='${}$'.format(gyro_labels[ii]))
        axs[3][0].plot(data_magnet[:,ii], label='${}$'.format(magnet_labels[ii]))

        axs[2][1].plot(data_gyro_inte[:,ii], label='${}$'.format(gyro_labels[ii]))

    for y in range(len(axs)):
        for x in range(len(axs[0])):
            if title_list[y][x] == '':
                continue
            axs[y][x].set_title(title_list[y][x],fontsize=20)
            axs[y][x].legend(fontsize=14)
            axs[y][x].set_ylabel(label_list[y][x],fontsize=18)

    fig.savefig('position_output.png',dpi=300,bbox_inches='tight',facecolor='#FCFCFC')
