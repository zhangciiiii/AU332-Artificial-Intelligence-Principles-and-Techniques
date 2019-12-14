import numpy as np
import quaternion

# Copy and paste here the first pose from the file groundtruth.txt
initial_pose = np.matrix('0.315971 -2.12211 1.36195 -0.222249 0.0102722 -0.298052 0.928259')

T_ros = np.matrix('-1 0 0 0;\
                    0 0 1 0;\
                    0 1 0 0;\
                    0 0 0 1')

T_m = np.matrix('1.0157    0.1828   -0.2389    0.0113;\
                 0.0009   -0.8431   -0.6413   -0.0098;\
                -0.3009    0.6147   -0.8085    0.0111;\
                      0         0         0    1.0000')

t = initial_pose[0,0:3].transpose()
q = np.quaternion(initial_pose[0,6],initial_pose[0,3],initial_pose[0,4],initial_pose[0,5])

R = quaternion.as_rotation_matrix(q)

T_0 = np.block([[R,t],[0, 0, 0, 1]])

T_g = T_ros * T_0 * T_ros * T_m

print(T_g)
print(np.linalg.inv(T_g))