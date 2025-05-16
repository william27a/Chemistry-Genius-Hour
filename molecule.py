import numpy as np
import mujoco
from chemlib import Element
import math

from scipy.spatial.transform import Rotation as R # type: ignore
import scipy.spatial

global num_elements
num_elements = 0

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
            global num_elements

            line = [x for x in line.split(' ') if x != '']

            if len(line) == 4 and line[0] == "dipole":
                self.dipole = (float(line[1]), float(line[2]), float(line[3]))
                return

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
                '<geom name=\"' + line[3] + '_' + str(num_elements) + '\" type=\"sphere\" pos=\"' + pos + '\" size=\"' + size + '\" rgba=\"' + rgba + '\" contype=\"0\" conaffinity=\"0\"/>'
            )

            num_elements += 1

        for line in fileText:
            addAtom(line)

        self.centerOfMass = [
            self.centerOfMass[0] / self.totalMass,
            self.centerOfMass[1] / self.totalMass,
            self.centerOfMass[2] / self.totalMass,
        ]

        # print(self.centerOfMass)

        self.appliedForce = (0, 0, 0)

    def assignPhysics(self, model, data, id):
        self.model = model
        self.data = data
        self.id = id

        self.robot_qpos_addr = self.id * 7
        self.robot_qvel_addr = self.id * 6

        # self.robot_qpos_addr = self.model.joint('robotfree' + str(id)).qposadr[0]

    def getElements(self):
        elements = []

        body_id = self.model.body(str(self.id)).id

        # Loop over all geoms and collect those with matching body id
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] == body_id:
                element = Element(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).split('_')[0])
                position = self.data.geom_xpos[geom_id]

                # print(element.properties)

                elements.append((position, element.properties['AtomicNumber']))
                # print(element.properties)
            
        return elements

    def getPosition(self):
        simPos = self.data.qpos[self.robot_qpos_addr: self.robot_qpos_addr+3]

        return [
            simPos[0] + self.centerOfMass[0],
            simPos[1] + self.centerOfMass[1],
            simPos[2] + self.centerOfMass[2],
        ]

    def getDipole(self):
        vector = np.array(self.dipole)

        if np.linalg.norm(vector) == 0: return None

        rotation = R.align_vectors([0, 0, -1.85], vector)[0]
        return rotation

    def getVelocity(self):
        return self.data.qvel[self.robot_qvel_addr: self.robot_qvel_addr+3]
    
    def getAcceleration(self):
        return self.data.qacc[self.robot_qvel_addr: self.robot_qvel_addr+3]

    def setPosition(self, x, y, z):
        self.data.qpos[self.robot_qpos_addr    ] = x - self.centerOfMass[0]  # x-axis position
        self.data.qpos[self.robot_qpos_addr + 1] = y - self.centerOfMass[1]  # y-axis position
        self.data.qpos[self.robot_qpos_addr + 2] = z - self.centerOfMass[2]  # z-axis position

    def setVelocity(self, dx, dy, dz):
        self.data.qvel[self.robot_qvel_addr    ] = dx  # x-axis velocity
        self.data.qvel[self.robot_qvel_addr + 1] = dy  # y-axis velocity
        self.data.qvel[self.robot_qvel_addr + 2] = dz  # z-axis velocity

        # # Set initial pose: rotate 90 degrees around Y-axis
        # rot = R.from_euler('z', 0, degrees=True)
        # quat = rot.as_quat()  # [x, y, z, w] format

        # # MuJoCo expects [w, x, y, z]
        # quat_mj = np.array([quat[3], quat[0], quat[1], quat[2]])

        # # Set orientation in d.qpos
        # self.data.qpos[self.robot_qpos_addr+3:self.robot_qpos_addr+7] = quat_mj

    def setAcceleration(self, ddx, ddy, ddz):
        dx, dy, dz = self.getVelocity()
        
        dx *= 0.50  # dampening
        dy *= 0.50  # dampening
        dz *= 0.50  # dampening

        self.data.qvel[self.robot_qvel_addr    ] = dx + (ddx * self.timestep) + ((np.random.rand() * 2 - 1) * 0.01)
        self.data.qvel[self.robot_qvel_addr + 1] = dy + (ddy * self.timestep) + ((np.random.rand() * 2 - 1) * 0.01)
        self.data.qvel[self.robot_qvel_addr + 2] = dz + (ddz * self.timestep) + ((np.random.rand() * 2 - 1) * 0.01)

    def applyForce(self, magnitude, otherMolecule):
        otherPos = otherMolecule.getPosition()
        selfPos = self.getPosition()

        dx, dy, dz = selfPos[0] - otherPos[0], selfPos[1] - otherPos[1], selfPos[2] - otherPos[2]
        
        deltaPositionMagnitude = ((dx ** 2) + (dy ** 2) + (dz ** 2)) ** 0.5
        dx, dy, dz = dx / deltaPositionMagnitude, dy / deltaPositionMagnitude, dz / deltaPositionMagnitude

        dx, dy, dz = dx * magnitude, dy * magnitude, dz * magnitude

        self.appliedForce = (
            self.appliedForce[0] + dx,
            self.appliedForce[1] + dy,
            self.appliedForce[2] + dz
        )

    def pointTo(self, molecules):
        dipole = self.getDipole()
        
        if dipole == None: return

        position = self.getPosition()

        rotations = []
        weights = []

        for molecule in molecules:
            if self == molecule: continue

            otherPosition = molecule.getPosition()
            deltaPosition = np.array(otherPosition) - np.array(position)

            rotations.append([
                (math.pi / 2) - math.atan(deltaPosition[1] / deltaPosition[0]),
                0,
                (math.pi / 2) - math.atan(deltaPosition[2] / deltaPosition[0])
            ])

            weights.append(np.linalg.norm(deltaPosition)**2)

        rotations = R.from_euler('xyz', rotations, False)

        targetRotation = rotations.mean(weights) * dipole
        
        self.data.qpos[self.robot_qpos_addr + 3: self.robot_qpos_addr + 7] = targetRotation.as_quat()

    def getBondStrength(self, molecule):
        currentMagnitude = np.linalg.norm(self.dipole)
        otherMagnitude = np.linalg.norm(molecule.dipole)

        currentDirection = R.from_quat(self.data.qpos[self.robot_qpos_addr + 3: self.robot_qpos_addr + 7])
        otherDirection = R.from_quat(molecule.data.qpos[molecule.robot_qpos_addr + 3: molecule.robot_qpos_addr + 7])

        currentVector = currentDirection.apply([0, 0, 1])
        otherVector = otherDirection.apply([0, 0, 1])

        # print(1 - scipy.spatial.distance.cosine(currentVector, otherVector))

        return (1 - scipy.spatial.distance.cosine(currentVector, otherVector)) * (1 - abs(currentMagnitude - otherMagnitude))

    def update(self):
        # print(self.appliedForce)
        self.setAcceleration(*self.appliedForce)
        self.appliedForce = (0, 0, 0, 0, 0, 0)

    def hasCollision(self):
        for contact in self.data.contact:
          if contact.geom1 != 0 and contact.geom2 != 0: # Check if the contact is valid
            return True
          
        return False