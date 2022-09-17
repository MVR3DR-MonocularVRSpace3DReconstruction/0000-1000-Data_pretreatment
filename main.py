
import time
import numpy as np
import csv,datetime
import matplotlib.pyplot as plt


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

    gyro_labels = ['w_x','w_y','w_z'] # gyro labels for plots
    acce_labels = ['a_x','a_y','a_z']
    magnet_labels = ['T_x','T_y','T_z']
    imu = read_trajectory("imu.txt")

    ###################################
    # Record New Data 
    ###################################

    data_acce_g = np.array([ii.pose[0] for ii in imu])
    data_acce_u = np.array([ii.pose[1] for ii in imu])
    data_gyro = np.array([ii.pose[2] for ii in imu]) # new values
    data_magnet = np.array([ii.pose[3] for ii in imu])

    ###################################
    # Integrate Acce Data
    ###################################

    









    ###################################
    # Plot with and without offsets
    ###################################
    
    plt.style.use('ggplot')
    fig,axs = plt.subplots(4,2,figsize=(24,18))

    for ii in range(0,3):
        axs[0][0].plot(data_acce_g[:,ii],
                    label='${}$'.format(acce_labels[ii]))
        axs[0][1].plot(data_acce_u[:,ii],
                    label='${}$'.format(acce_labels[ii]))
        axs[0][2].plot(data_gyro[:,ii],
                    label='${}$'.format(gyro_labels[ii]))
        axs[0][3].plot(data_magnet[:,ii],
                    label='${}$'.format(magnet_labels[ii]))

    axs[0][0].set_title('IMU device Odometry for Camera Extrinsic',fontsize=20)
    axs[0][0].legend(fontsize=14)
    axs[0][0].set_ylabel('$a_{x,y,z}$ [$m^{2}/s$]',fontsize=18)
    axs[0][0].set_ylim([-20,20])

    axs[0][1].legend(fontsize=14)
    axs[0][1].set_ylabel('$a_{x,y,z}$ [$m^{2}/s$]',fontsize=18)
    axs[0][1].set_ylim([-2,2])
                    
    axs[0][2].legend(fontsize=14)
    axs[0][2].set_ylabel('$w_{x,y,z}$ [$^{\circ}/s$]',fontsize=18)
    axs[0][2].set_ylim([-2,2])

    axs[0][3].legend(fontsize=14)
    axs[0][3].set_ylabel('$T_{x,y,z}$',fontsize=18)
    axs[0][3].set_ylim([-50,50])
    

    # axs[0].set_title('IMU device Odometry for Camera Extrinsic',fontsize=22)

    fig.savefig('gyro_calibration_output.png',dpi=300,bbox_inches='tight',facecolor='#FCFCFC')
    # fig.show()
    # input()