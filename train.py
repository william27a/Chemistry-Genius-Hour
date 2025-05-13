import time

import mujoco
import mujoco.viewer

import contextlib
import signal

from molecule import Molecule

import numpy as np
import torch

import os

# class TimeoutException(Exception): pass

# @contextlib.contextmanager
# def time_limit(seconds):
#     def signal_handler(signum, frame):
#         raise TimeoutException("Timed out!")
#     signal.signal(signal.SIGALRM, signal_handler)
#     signal.alarm(seconds)
#     try:
#         yield
#     finally:
#         signal.alarm(0)

def simulate(molecules):
  def getRadius(element1, element2):
    dx = element1[0] - element2[0]
    dy = element1[1] - element2[1]
    dz = element1[2] - element2[2]

    diameter = ((dx ** 2) + (dy ** 2) + (dz ** 2)) ** 0.5
    radius = diameter / 2

    return radius

  def getForce(radius):
    proton_charge = 1.602
    coulombs_constant = 8.98 * 10 # *10 goes from meters to angstroms (already factored in the magnitude of a proton 10^-19 * 10^-19 * 10^29)
    return proton_charge * proton_charge * coulombs_constant / (radius * radius)

  def repulse(molecule1, molecule2):
    forceMagnitude = 0.0

    
    for element1 in molecule1.getElements():
      for element2 in molecule2.getElements():
        radius = getRadius(element1, element2)
        forceMagnitude += getForce(radius)

    molecule1.applyForce(forceMagnitude, molecule2)
    molecule2.applyForce(forceMagnitude, molecule1)

  n = len(molecules)

  for i in range(0, n):
    for j in range(i+1, n):
      repulse(molecules[i], molecules[j])


def runTrainingLoop(xml, hyperparameters):
  def launchViewer(m, d, render):
    if render:
      return mujoco.viewer.launch_passive(m, d)
    return contextlib.nullcontext()
  
  try:
    molecules = []

    files = os.listdir('./molecules')

    with open(xml, 'r') as model:
      modelLines = model.readlines()

    insertIndex = None
    for index, line in enumerate(modelLines):

      if 'insert here' in line:
        insertIndex = index
        break

    modelLines.insert(insertIndex, '')

    for file in files:
      with open('./molecules/' + file, 'r') as file:
        fileText = [line.strip() for line in file.readlines()]

        molecule = []

        molecules.append(Molecule(molecule, fileText, 1 / hyperparameters["fps"]))

        modelLines.insert(insertIndex, '        </body>\n\n')

        for geom in molecule:
          modelLines.insert(insertIndex, '            '+geom+'\n')

        modelLines.insert(insertIndex, '            <joint type="free" name="robotfree' + str(len(molecules)-1) + '"/>\n')
        modelLines.insert(insertIndex, '        <body pos=\"0 0 ' + str(len(molecules)-1) + '\" name=\"' + str(len(molecules)-1) + '\">\n')

    with open('modelDuplicate.xml', 'w') as out:
      for line in modelLines:
        out.write(line)

    m = mujoco.MjModel.from_xml_path('modelDuplicate.xml')
    m.opt.timestep = 1 / hyperparameters["fps"]
    # max_steps = max_time / m.opt.timestep
    d = mujoco.MjData(m)

    for index, molecule in enumerate(molecules):
      molecule.assignPhysics(m, d, index)

    # robot = Robot(m, d, hyperparameters["visualize"])

    with launchViewer(m, d, hyperparameters["render"]) as viewer:
      step_start = time.time()

      if viewer:
      # #   # Access the camera object
        cam = viewer.cam

        cam.azimuth = 30
        cam.elevation = -30
        cam.distance = 100
        cam.lookat[:] = [20, 0, 0]  # what the camera is looking at

      while (not viewer or viewer.is_running()):
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.

        # simulate(molecules)
        # for molecule in molecules:
          # print(molecule.appliedForce)
          # molecule.update()

        simulate(molecules)
        for molecule in molecules:
          molecule.update()

        mujoco.mj_step(m, d)

        # write code here

        # simulate(molecules)
        # for molecule in molecules:
        #   molecule.update()
          # molecule.setVelocity(0, 0, 0.5)

        mujoco.mj_forward(m, d)  # Update simulation with new state

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        if hyperparameters["render"]:
          viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.

        # time_until_next_step = m.opt.timestep - (time.time() - step_start)
        # if viewer and time_until_next_step > 0:
          # time.sleep(time_until_next_step)

      
      mujoco.mj_resetData(m, d)

  except KeyboardInterrupt:
    # torch.save(model.state_dict(), 'model')
    return
  
  # torch.save(model.state_dict(), 'model')

if __name__ == "__main__":
  hyperparameters = {
    "render": True,
    "fps": 100000
  }

  runTrainingLoop("model.xml", hyperparameters)

  # try:
  #     with time_limit(60):
  #         runTrainingLoop("model.xml", hyperparameters)
  # except TimeoutException as e:
  #     print("Timed out!")