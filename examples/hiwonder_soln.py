import sys
import os

sys.path.append(os.path.abspath("examples"))

import math
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import FiveDOFRobotTemplate
from traj_gen import CubicPolynomial



class FiveDOFRobot(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()
    

    def calc_forward_kinematics(self, joint_values: list, radians=True):
        """
        Calculate forward kinematics based on the provided joint angles.
        
        Args:
            theta: List of joint angles (in degrees or radians).
            radians: Boolean flag to indicate if input angles are in radians.
        """
        curr_joint_values = joint_values.copy()
        
        if not radians: # Convert degrees to radians if the input is in degrees
            curr_joint_values = [np.deg2rad(theta) for theta in curr_joint_values]
        
        # Ensure that the joint angles respect the joint limits
        for i, theta in enumerate(curr_joint_values):
            curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])

        # Set the Denavit-Hartenberg parameters for each joint
        DH = np.zeros((self.num_dof, 4)) # [theta, d, a, alpha]
        DH[0] = [curr_joint_values[0], self.l1, 0, -np.pi/2]
        DH[1] = [curr_joint_values[1] - np.pi/2, 0, self.l2, np.pi]
        DH[2] = [curr_joint_values[2], 0, self.l3, np.pi]
        DH[3] = [curr_joint_values[3] + np.pi/2, 0, 0, np.pi/2]
        DH[4] = [curr_joint_values[4], self.l4 + self.l5, 0, 0]

        # Compute the transformation matrices
        Hlist = [ut.dh_to_matrix(dh) for dh in DH]

        # Precompute cumulative transformations to avoid redundant calculations
        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ Hlist[i])

        # Calculate EE position and rotation
        H_ee = H_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist
    

    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt=0.02):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy, vz].
        """
        new_joint_values = joint_values.copy()

        # move robot slightly out of zeros singularity
        if all(theta == 0.0 for theta in new_joint_values):
            new_joint_values = [theta + np.random.rand()*0.02 for theta in new_joint_values]
        
        # Calculate the joint velocity using the inverse Jacobian
        joint_vel = self.inverse_jacobian(new_joint_values, pseudo=True) @ vel
        # joint_vel = self.damped_inverse_jacobian(new_joint_values) @ vel

        joint_vel = np.clip(joint_vel, 
                            [limit[0] for limit in self.joint_vel_limits], 
                            [limit[1] for limit in self.joint_vel_limits]
                        )

        # Update the joint angles based on the velocity
        for i in range(self.num_dof):
            new_joint_values[i] += dt * joint_vel[i]

        # Ensure joint angles stay within limits
        new_joint_values = np.clip(new_joint_values, 
                               [limit[0] for limit in self.joint_limits], 
                               [limit[1] for limit in self.joint_limits]
                            )
        
        return new_joint_values
    

    def jacobian(self, joint_values: list):
        """
        Compute the Jacobian matrix for the current robot configuration.

        Args:
            joint_values (list): The joint angles for the robot.

        Returns:
            Jacobian matrix (3x5).
        """
        _, Hlist = self.calc_forward_kinematics(joint_values)

        # Precompute transformation matrices for efficiency
        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ Hlist[i])

        # Define O0 for calculations
        O0 = np.array([0, 0, 0, 1])
        
        # Initialize the Jacobian matrix
        jacobian = np.zeros((3, self.num_dof))

        # Calculate the Jacobian columns
        for i in range(self.num_dof):
            H_curr = H_cumulative[i]
            H_final = H_cumulative[-1]
            
            # Calculate position vector r
            r = (H_final @ O0 - H_curr @ O0)[:3]

            # Compute the rotation axis z
            z = H_curr[:3, :3] @ np.array([0, 0, 1])

            # Compute linear velocity part of the Jacobian
            jacobian[:, i] = np.cross(z, r)

        # Replace near-zero values with zero, primarily for debugging purposes
        return ut.near_zero(jacobian)
    

    def jacobian6x5(self, joint_values: list = None):
        """
        Compute the Jacobian matrix for the current robot configuration.

        Args:
            theta (list, optional): The joint angles for the robot. Defaults to self.theta.
        
        Returns:
            Jacobian matrix (6x5).
        """
        _, Hlist = self.calc_forward_kinematics(joint_values)

        # Precompute transformation matrices for efficiency
        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ Hlist[i])

        # Define O0 for calculations
        O0 = np.array([0, 0, 0, 1])
        
        # Initialize the Jacobian matrix
        jacobian = np.zeros((6, self.num_dof))

        # Calculate the Jacobian columns
        for i in range(self.num_dof):
            H_curr = H_cumulative[i]
            H_final = H_cumulative[-1]
            
            # Calculate position vector r
            r = (H_final @ O0 - H_curr @ O0)[:3]

            # Compute the rotation axis z
            z = H_curr[:3, :3] @ np.array([0, 0, 1])

            # Compute linear velocity part of the Jacobian
            jacobian[:3, i] = np.cross(z, r)

            # Compute angular velocity part of the Jacobian
            jacobian[3:, i] = z

        # Replace near-zero values with zero, primarily for debugging purposes
        return ut.near_zero(jacobian)
  

    def inverse_jacobian(self, joint_values: list, pseudo=False):
        """
        Compute the inverse of the Jacobian matrix using either pseudo-inverse or regular inverse.
        
        Args:
            pseudo: Boolean flag to use pseudo-inverse (default is False).
        
        Returns:
            The inverse (or pseudo-inverse) of the Jacobian matrix.
        """

        J = self.jacobian(joint_values)
        JT = np.transpose(J)
        manipulability_idx = np.sqrt(np.linalg.det(J @ JT))
        # print(f'Manipulability index is: {manipulability_idx:.03f}')

        # ev = np.linalg.eigvals(JJT)
        # print(ev)

        if pseudo:
            return np.linalg.pinv(self.jacobian(joint_values))
        else:
            return np.linalg.inv(self.jacobian(joint_values))
        
        
    def damped_inverse_jacobian(self, joint_values: list, damping_factor=0.025):
        
        J = self.jacobian(joint_values)
        # print(f'jacobian3x5: \n {self.jacobian(q)}')
        JT = np.transpose(J)
        I = np.eye(3)
        return JT @ np.linalg.inv(J @ JT + (damping_factor**2)*I)


    def calc_inverse_kinematics(self, ee: ut.EndEffector, joint_values: list, soln: int=0) -> list:
        # Extract position and orientation of the end-effector
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
        
        # Desired position and rotation matrix
        p = np.array([ee.x, ee.y, ee.z])
        rpy = np.array([ee.rotx, ee.roty, ee.rotz])
        R = ut.euler_to_rotm(rpy)
        
        # Calculate wrist position
        wrist_pos = p - R @ np.array([0, 0, 1]) * (l4 + l5)
        # print(f"Wrist pose: \n {wrist_pos} \n")
        # print(f"Rotation matrix, R: {R}, \n Determinant: {np.linalg.det(R)}")

        try: 
            # Solve for theta_1 using trigonometry
            x_wrist, y_wrist, z_wrist = wrist_pos
            theta1 = [math.atan2(y_wrist, x_wrist), math.atan2(-y_wrist, -x_wrist)]
            # TIP #1: theta1 has two solutions (+ve and -ve) wrapped around -pi and pi

            # using cosine rule, find theta_3
            s = z_wrist - l1
            r = [-math.sqrt(x_wrist**2 + y_wrist**2), math.sqrt(x_wrist**2 + y_wrist**2)]
            # TIP #2: consider two cases for r (+ve and -ve)

            L = math.sqrt(r[0]**2 + s**2)
            ctheta3 = (L**2 - l2**2 - l3**2) / (2 * l2 * l3)
            ctheta3 = max(-1.0, min(1.0, ctheta3))

            theta3 = [math.atan2(math.sqrt(1 - ctheta3**2), ctheta3), math.atan2(-math.sqrt(1 - ctheta3**2), ctheta3)]
            # TIP #3: Alternative approach to finding theta3
            # beta = np.arccos((-L**2 + l2**2 + l3**2) / (2 * l2 * l3))
            # theta3 = [pi - beta, beta - pi]   

            possible_solns, valid_solns= [], []

            for th3 in theta3:
                # calculate theta2
                # theta2 = [-(math.atan2(r[0], s) - math.atan2(l3*math.sin(th3), l2 + l3*math.cos(th3))), \
                #           -(math.atan2(r[1], s) - math.atan2(l3*math.sin(th3), l2 + l3*math.cos(th3)))]
                theta2 = [-ut.wraptopi(np.pi/2 - math.atan2(s, r[0]) - math.atan2(l3*math.sin(th3), l2 + l3*math.cos(th3))), \
                          -ut.wraptopi(np.pi/2 - math.atan2(s, r[1]) - math.atan2(l3*math.sin(th3), l2 + l3*math.cos(th3)))]
                # TIP #4: we use the two values of r to compute two values of theta2

                # print(f'theta2: {[float(round(np.rad2deg(th),2)) for th in theta2]}') 
                
                # calculate theta4 and theta5
                for th2 in theta2:
                    c23_ = math.cos(th2)*math.cos(th3) + math.sin(th2)*math.sin(th3)
                    s23_ = math.sin(th2)*math.cos(th3) - math.cos(th2)*math.sin(th3)

                    # th1 = theta1
                    for th1 in theta1:
                        c1, s1 = math.cos(th1), math.sin(th1)
                        R0_3 = np.array([[c1*s23_, c1*c23_, s1],
                                        [s1*s23_, s1*c23_, -c1],
                                        [c23_, -s23_, 0]])
                        # TIP #5 This is derived from calculating R3_4*R4_5 from the DH table
                        R3_5 = R0_3.T @ R

                        th4 = math.atan2(R3_5[1,2], R3_5[0,2])
                        th5 = math.atan2(-R3_5[2,0], -R3_5[2,1])

                        solution = [th1, th2, th3, th4, th5]
                        possible_solns.append(solution)

                        if ut.check_valid_ik_soln(solution, ee, self):
                            valid_solns.append(solution)

            print(f"\nPossible solutions: ")
            for solution in possible_solns:
                print(f"{[float(round(np.rad2deg(th),2)) for th in solution]}")
            print(f"\nValid solutions: ")
            for solution in valid_solns:
                print(f"{[float(round(np.rad2deg(th),2)) for th in solution]}")

            if len(valid_solns) > 0:
                if len(valid_solns) > 1:
                    if soln == 0:
                        new_joint_values = valid_solns[0]
                    elif soln == 1:
                        new_joint_values = valid_solns[1]
                else:
                    new_joint_values = valid_solns[0]
                return new_joint_values
            else:
                raise ValueError(f"\n[ERROR] No valid solution found! Joint limits may be exceeded or position may be unreachable.")

        except RuntimeWarning:
            print("\n [ERROR] Joint limits exceeded! \n")


    def calc_numerical_ik(self, ee: ut.EndEffector, joint_values: list, tol = 0.01, ilimit = 500):

        new_joint_values = joint_values.copy()
        
        # move robot slightly out of zeros singularity
        if all(theta == 0.0 for theta in new_joint_values):
            new_joint_values = [theta + np.random.rand()*0.02 for theta in new_joint_values]

        q = new_joint_values
        i = 0
        print(f"\nInitial value at start = {q} \n")

        while i < ilimit:
            i += 1

            # compute current EE position based on q
            ee_current, _ = self.calc_forward_kinematics(q, radians=True)

            # calculate the EE position error
            e = self.calc_error(ee, ee_current)

            # update q
            J = self.jacobian(q)
            # q += np.linalg.pinv(J) @ e
            q += self.damped_inverse_jacobian(q) @ e

            # check for joint limits
            for j, th in enumerate(q):
                q[j] = np.clip(th, self.joint_limits[j][0], self.joint_limits[j][1])

            print(f"iterating... {i}/{ilimit} | q = {q}")

            # Check if we have arrived
            if abs(max(e, key=abs)) < tol:
                print(f"\nSolution found = {q} | Max pos error = {max(e, key=abs)} | # iterations: {i}/{ilimit}  \n")
                return q
        
        if abs(max(e, key=abs)) > tol:
            print("\n [ERROR] Numerical IK solution failed to converge... \n \
                Possible causes: \n \
                1. joint limits may have been exceeded! \n \
                2. desired joint configuration is very close to OR at a singularity \n")
            print(f"Max position error: {max(e, key=abs)} | # iterations: {i}/{ilimit} ")
            # raise ValueError
            return None


    def calc_numerical_ik_restarts(self, ee: ut.EndEffector, joint_values: list, tol = 0.002, ilimit = 100, slimit = 10):
       
        k = 0

        while k < slimit:
            i = 0
            q = ut.sample_valid_joints(self) # randomly samples a new valid joint configuration
            k += 1

            while i < ilimit:
                i += 1

                # compute current EE position based on q
                ee_current, _ = self.calc_forward_kinematics(q, radians=True)

                # calculate the EE position error
                e = self.calc_error(ee, ee_current)

                # update q
                J = self.jacobian(q)
                q += np.linalg.pinv(J) @ e
                # q += self.damped_inverse_jacobian(q) @ e

                # check for joint limits
                for j, th in enumerate(q):
                    q[j] = np.clip(th, self.joint_limits[j][0], self.joint_limits[j][1])

                print(f"Start {k} | iterating... {i}/{ilimit} | q = {q}")

                # Check if we have arrived
                if abs(max(e, key=abs)) < tol:
                    print(f"\nSolution found = {q} | Max pos error = {max(e, key=abs)} | # iterations: {i}/{ilimit}  \n")
                    return q
            
            print(f"Start {k} failed, Max position error: {max(e, key=abs)} | # iterations: {i}/{ilimit} ")

        print("\n [ERROR] Numerical IK solution failed to converge... \n \
            Possible causes: \n \
            1. joint limits may have been exceeded! \n \
            2. desired joint configuration is very close to OR at a singularity \n")
        print(f"Max position error: {max(e, key=abs)} | # iterations: {i}/{ilimit} ")
        # raise ValueError
        return None


    def calc_error(self, ee_desired: ut.EndEffector, ee_current: ut.EndEffector) -> ut.EndEffector:
        return [ee_desired.x - ee_current.x,
                ee_desired.y - ee_current.y,
                ee_desired.z - ee_current.z
                ]



if __name__ == "__main__":
    
    robot_model = FiveDOFRobot()
    traj_model = CubicPolynomial()
    
    robot = RobotSim(robot_model=robot_model, traj_model=traj_model)
    viz = Visualizer(robot=robot)
    viz.run()