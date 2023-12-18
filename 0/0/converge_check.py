#!/usr/bin/env python3

import sys,os
sys.path.append(sys.path[0] + '/../../Utilities')
import numpy as np
import VASP_classes as VC

# If we want to do this for all subdirectories as well, then just add the -r tag

def main():
	
	if len(sys.argv) > 1 and sys.argv[1] == '-r':
		# Find all subdirectories
		for filename in os.listdir():
			if os.path.isdir(filename):
				print(f'>>> {filename}/')
				if not os.path.isfile(f'{filename}/OUTCAR'):
					print(f'Could not find OUTCAR.')

				elif not os.path.isfile(f'{filename}/POSCAR'):
					print(f'Could not find POSCAR.')
				
				else:
					
					outcar = VC.OUTCAR(f'{filename}/OUTCAR')
					poscar = VC.POSCAR(f'{filename}/POSCAR')
					generate_report(outcar=outcar,poscar=poscar)
					print('')

	else:
		if not os.path.isfile('OUTCAR'):
			print('Could not find OUTCAR.')
			exit(1)

		elif not os.path.isfile('POSCAR'):
			print('Could not find POSCAR.')
			exit(1)

		else:
			outcar = VC.OUTCAR('OUTCAR')
			poscar = VC.POSCAR('POSCAR')
			generate_report(outcar=outcar,poscar=poscar)

def generate_report(outcar,poscar):
	# See if any ionic steps have completed - if not, break out
	if outcar.ionic_steps == 0:
		print('No ionic steps completed')
		return

	if poscar.selective_dynamics:
		# Check to see which atoms are included in the force optimizer
		relaxed_atoms = np.array([np.any(i == 'T') for i in poscar.fixation_data])
	else:
		# If there's no selective dynamics, take all atoms as relaxed
		relaxed_atoms = np.array([True for i in poscar.pos_data])

	# If the job is not an NEB calculation (those need different forces)
	if not outcar.is_neb:
		# Get the magnitude relaxed forces for all ionic steps
		mag_forces = np.linalg.norm(outcar.forces[:,relaxed_atoms,:],axis=2)

		# The max relaxed force magnitude of each ionic step
		max_forces = np.array([max(ionic_step_mag_forces) for ionic_step_mag_forces in mag_forces])

		# Print out all of the data
		print('Ionic Step   E(sg->0) eV   max|F_relaxed| eV/A   avg|F_relaxed| eV/A')
		for step,data in enumerate(zip(outcar.e_sigzero,max_forces,mag_forces)):
			print(f'{step+1:10}   {data[0]:11.5f}   {data[1]:19.5f}   {np.average(data[2]):19.5f}')
	
	# If it is an NEB and tangent forces has something written out to it, get the tangent forces
	elif outcar.is_neb and len(outcar.tangent_vectors) > 0:
		# Get the relaxed tangent forces for all ionic steps
		
		tangent_forces = []
		for ionic_step in zip(outcar.forces[:,relaxed_atoms,:],outcar.tangent_vectors[:,relaxed_atoms,:]):
			ionic_tangent_forces = []
			for atom in zip(ionic_step[0],ionic_step[1]):
				atom_tangent_forces = np.dot(atom[0],atom[1]) * atom[1]
				ionic_tangent_forces.append(atom_tangent_forces)
			
			tangent_forces.append(ionic_tangent_forces)

		tangent_forces = np.array(tangent_forces)

		perp_forces = outcar.forces[:,relaxed_atoms,:] - tangent_forces
		mag_perp_forces = np.linalg.norm(perp_forces,axis=2)
		max_perp_forces =  np.array([max(ionic_step_mag_perp_forces) for ionic_step_mag_perp_forces in mag_perp_forces])

		# Print out all of the data
		print('Ionic Step   E(sg->0) eV   max|F_perp_relaxed| eV/A   avg|F_perp_relaxed| eV/A')
		for step,data in enumerate(zip(outcar.e_sigzero,max_perp_forces,mag_perp_forces)):
			print(f'{step+1:10}   {data[0]:11.5f}   {data[1]:24.5f}   {np.average(data[2]):24.5f}')
	
	# If an OUTCAR was found but no forces were written out to it, tell the user
	else:
		print('Could not find any forces.')

if __name__ == '__main__':
	main()