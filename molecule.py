import numpy as np
import mujoco
import torch
from chemlib import Element

from scipy.spatial.transform import Rotation as R # type: ignore

def getMolecularColor(abbreviatedName):
    colors = {
        ('H'): '1 1 1 1',
        ('C'): '0 0 0 1',
        ('N'): '0 0 1 1',
        ('O'): '1 0 0 1',
        ('F', 'Cl'): '0 0.5 0 1',
        ('Br'): '0.54 0 0 1',
        ('I'): '0.58 0 0.82 1',
        ('He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'): '0 1 1 1',
        ('P'): '1 0.65 0 1',
        ('S'): '1 1 0 1',
        ('B'): '0.96 0.96 0.86 1',
        ('Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'): '0.89 0.50 0.93 1',
        ('Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'): '0 1 0.39 1',
        ('Ti'): '0.5 0.5 0.5 1',
        ('Fe'): '1 0.55 0 1'
    }

    for names, color in colors.items():
        if abbreviatedName in names:
            return color
        
    return '1 0.76 0.79 1'

class Molecule():
    def __init__(self, molecule, fileText, timestep):
        self.timestep = timestep

        self.centerOfMass = [0, 0, 0]
        self.totalMass = 0

        def addAtom(line):
            line = [x for x in line.split(' ') if x != '']

            if len(line) < 4 or line[3][0] not in 'QWERTYUIOPASDFGHJKLZXCVBNM':
                return
            
            pos = ' '.join(line[:3])
            size = '0.529 0.529 0.529' # orbital radius of first electrons in Angstroms, the unit for the .mol file (the nucleus radius is wayyy to small)
            if line[3] == 'H':
                size = '0.1 0.1 0.1'
            rgba = getMolecularColor(line[3])

            element = Element(line[3])
            atomicMass = element.properties['AtomicMass']
            self.totalMass += atomicMass

            self.centerOfMass = [
                self.centerOfMass[0] + (float(line[0]) * atomicMass),
                self.centerOfMass[1] + (float(line[1]) * atomicMass),
                self.centerOfMass[2] + (float(line[2]) * atomicMass),
            ]

            # print(element.properties['AtomicMass'])

            molecule.append(
                '<geom type=\"sphere\" pos=\"' + pos + '\" size=\"' + size + '\" rgba=\"' + rgba + '\"/>'
            )

        for line in fileText:
            addAtom(line)

        self.centerOfMass = [
            self.centerOfMass[0] / self.totalMass,
            self.centerOfMass[1] / self.totalMass,
            self.centerOfMass[2] / self.totalMass,
        ]

        # print(self.centerOfMass)

        self.appliedForce = (0, 0, 0)

        # print(molecule)

        # self.model = model
        # self.data = data

        # self.robot_qpos_addr = self.model.joint('robotfree' + str(id)).qposadr[0]

        # self.predictedDistances = [
        #     0,
        #     0,
        #     0,
        #     0
        # ]

        # self.alpha = 0.91

        # self.visualize = visualize
        # if self.visualize:
        #     self.physicalRays = [
        #         self.model.geom('forward'),
        #         self.model.geom('right'),
        #         self.model.geom('backward'),
        #         self.model.geom('left'),
        #     ]

    def assignPhysics(self, model, data, id):
        self.model = model
        self.data = data
        self.id = id

        self.robot_qpos_addr = self.model.joint('robotfree' + str(id)).qposadr[0]

    def getElements(self):
        elements = []

        body_id = self.model.body(str(self.id)).id

        # Loop over all geoms and collect those with matching body id
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] == body_id:
                elements.append(self.data.geom_xpos[geom_id])
            
        return elements

    def getState(self, deltaPosition):
        state = deltaPosition

        # distances = self.getDistances()
        # realDistances = [0] * len(distances)

        # if self.visualize:
        #     self.alterPhysicalRays(distances)

        # for i, distance in enumerate(distances):
        #     if distance[1] != -1:
        #         realDistances[i] = distance[0]
        
        # self.updatePredictedDistances(realDistances)

        # for distance in self.predictedDistances:
        #     # maximum acceleration parameter, 1 is 100% or 144 inches
        #     # if the robot is 10 inches from a wall, accelerate in at
        #     max_accel = 4.0 / 144

        #     optimalDistance = 30
        #     distance = distance - optimalDistance

        #     # state.append(min(max(1 / max(distance, 1e-6), -max_accel), max_accel))
        #     state.append(2 * max_accel * distance / ((distance * distance) + 1))

        # velocity = self.getVelocity()
        # velocity[0] = velocity[0] / 100
        # velocity[1] = velocity[1] / 100
        # state.extend(velocity)
        # # state.extend([0, 0])
        # state = torch.Tensor(state)
        return state

    def getPosition(self):
        simPos = self.data.qpos[self.robot_qpos_addr: self.robot_qpos_addr+3]
        # print(simPos)

        return [
            simPos[0] + self.centerOfMass[0],
            simPos[1] + self.centerOfMass[1],
            simPos[2] + self.centerOfMass[2],
        ]

    def getVelocity(self):
        return self.data.qvel[self.robot_qpos_addr: self.robot_qpos_addr+3]
    
    def getAcceleration(self):
        return self.data.qacc[self.robot_qpos_addr: self.robot_qpos_addr+3]

    def setPosition(self, x, y):
        self.data.qpos[self.robot_qpos_addr    ] = x  # x-axis position
        self.data.qpos[self.robot_qpos_addr + 1] = y  # y-axis position

    def setVelocity(self, dx, dy, dz):
        self.data.qvel[self.robot_qpos_addr    ] = dx  # x-axis velocity
        self.data.qvel[self.robot_qpos_addr + 1] = dy  # y-axis velocity
        self.data.qvel[self.robot_qpos_addr + 2] = dz  # z-axis velocity

        # Set initial pose: rotate 90 degrees around Y-axis
        # rot = R.from_euler('z', 0, degrees=True)
        # quat = rot.as_quat()  # [x, y, z, w] format

        # # MuJoCo expects [w, x, y, z]
        # quat_mj = np.array([quat[3], quat[0], quat[1], quat[2]])

        # # Set orientation in d.qpos
        # self.data.qpos[self.robot_qpos_addr+3:self.robot_qpos_addr+7] = quat_mj

    def setAcceleration(self, ddx, ddy, ddz):
        dx, dy, dz = self.getVelocity()

        self.data.qvel[self.robot_qpos_addr    ] = dx + (ddx * self.timestep)
        self.data.qvel[self.robot_qpos_addr + 1] = dy + (ddy * self.timestep)
        self.data.qvel[self.robot_qpos_addr + 2] = dz + (ddz * self.timestep)

        # self.data.qacc[self.robot_qpos_addr    ] = ddx * 1  # x-axis acceleration
        # self.data.qacc[self.robot_qpos_addr + 1] = ddy * 1  # y-axis acceleration
        # self.data.qacc[self.robot_qpos_addr + 2] = ddz * 1  # z-axis acceleration
        # print(self.data.qvel[self.robot_qpos_addr: self.robot_qpos_addr + 3])

    def applyForce(self, magnitude, otherMolecule):
        otherPos = otherMolecule.getPosition()
        selfPos = self.getPosition()

        # print(selfPos)

        dx, dy, dz = selfPos[0] - otherPos[0], selfPos[1] - otherPos[1], selfPos[2] - otherPos[2]
        
        deltaPositionMagnitude = ((dx ** 2) + (dy ** 2) + (dz ** 2)) ** 0.5
        dx, dy, dz = dx / deltaPositionMagnitude, dy / deltaPositionMagnitude, dz / deltaPositionMagnitude

        dx, dy, dz = dx * magnitude, dy * magnitude, dz * magnitude

        self.appliedForce = (
            self.appliedForce[0] + dx,
            self.appliedForce[1] + dy,
            self.appliedForce[2] + dz,
        )

    def update(self):
        self.setVelocity(*self.appliedForce)
        self.appliedForce = (0, 0, 0)

    def hasCollision(self):
        for contact in self.data.contact:
          if contact.geom1 != 0 and contact.geom2 != 0: # Check if the contact is valid
            return True
          
        return False