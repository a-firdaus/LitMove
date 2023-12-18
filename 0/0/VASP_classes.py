#!/usr/bin/env python3

import numpy as np
import scipy.interpolate as si
import os
from copy import deepcopy

class POSCAR:
	def __init__(self,filepath,*,is_str_not_file=False):
		
		if is_str_not_file:
			self.file_data = [i.split() for i in filepath.splitlines()]
		else:
			with open(filepath,'r') as f:
				self.file_data = [i.split() for i in f.read().splitlines()]
		
		# Check to see if the file has the 0s left over from CONTCARs - this section removes the break characters and zeros at the end of the file until real data is seen again
		for revline in reversed(self.file_data):
			if len(revline) == 0 or revline[0] == '0.00000000E+00' or revline[0] == 'NaN':
				self.file_data.pop(-1)
			else:
				break

		self.name = ' '.join(self.file_data[0])

		# Get the lattice parameter data
		self.lattice_scalar = float(self.file_data[1][0])
		self.factored_lattice_parameters = np.array(self.file_data[2:5]).astype(np.float64)
		self.lattice_parameters = self.lattice_scalar * self.factored_lattice_parameters
		self.volume = np.linalg.det(self.lattice_parameters)

		# Get atom header data
		self.atom_header = [self.file_data[5],[int(i) for i in self.file_data[6]]]
		self.atom_list = []
		for speciesnum,species in enumerate(self.atom_header[0]):
			for occur in range(self.atom_header[1][speciesnum]):
				self.atom_list.append(species)

		self.atom_list = np.array(self.atom_list)

		# Check whether the POSCAR has selective dynamics on
		startstr = ' '.join(self.file_data[7])[0]
		if startstr == 's' or startstr == 'S': # If the first letter is s or S its selective dynamics
			self.selective_dynamics = True
		else:
			self.selective_dynamics = False

		# If there is sel. dyn. then the atom position type (cart or direct) will be on the
		if self.selective_dynamics:
			startstr = ' '.join(self.file_data[8])[0]
		else:
			startstr = ' '.join(self.file_data[7])[0]
		if startstr == 'd' or startstr == 'D':
			self.coordinate_type = 'dir'
		else:
			self.coordinate_type = 'cart'
		
		# Get atomic positions and fixation data if it exists
		if self.selective_dynamics:
			self.pos_data = np.array(self.file_data[9:])[:,:3].astype(np.float64)
			self.fixation_data = np.array(self.file_data[9:])[:,3:]
		else:
			self.pos_data = np.array(self.file_data[8:]).astype(np.float64)
			self.fixation_data = None

	# Return the cartesian coordinates for the atoms
	def get_cart_positions(self):
		if not self.coordinate_type == 'cart':
			return np.matmul((self.lattice_scalar*self.factored_lattice_parameters),self.pos_data.T).T
		else:
			return self.pos_data
	
	# Return the fractional coordinates for the atoms
	def get_dir_positions(self):
		if not self.coordinate_type == 'dir':
			return np.matmul(np.linalg.inv(self.lattice_scalar*self.factored_lattice_parameters),self.pos_data.T).T
		else:
			return self.pos_data


	# Refresh the atom header based on the curret atomlist
	def remake_atom_header(self):
		temp_atom_header = []
		current_species = ''
		current_species_count = 0
		for atom in self.atom_list:
			if current_species != atom:
				if current_species != '':
					temp_atom_header.append([current_species,current_species_count])
				current_species = atom
				current_species_count = 1
			else:
				current_species_count += 1

		temp_atom_header.append([current_species,current_species_count])
		self.atom_header = np.array(temp_atom_header).T.tolist()

	# Return the index of the atom with the coordinates closes to coord (a list of x,y,z)
	def find_closest_atom(self,coord):
		sse = np.asarray(self.pos_data).astype(float)
		for axis in range(3):
			sse[:,axis] = np.square(sse[:,axis] - coord[axis])

		return (sse[:,0] + sse[:,1] + sse[:,2]).argmin()

	# Fix all of the atoms above a specific c cutoff coordinate
	def affix_above(self,cutoff):
		# If selective dynamics was not enabled, copy over the pos_data array to the fixation_data (has the same shape as we need but will be overwitten)
		if not self.selective_dynamics:
			self.selective_dynamics = True
			self.fixation_data = deepcopy(self.pos_data.astype(str))
		
		for rownum, row in enumerate(self.fixation_data):
			for colnum, col in enumerate(row):
				if self.pos_data[rownum][2] > cutoff:
					self.fixation_data[rownum][colnum] = 'T'
				else:
					self.fixation_data[rownum][colnum] = 'F'

	# Creates supercells of the current lattice - this operation can only be performed once
	def replicate(self,xrep,yrep,zrep):
		
		# Because the math is incredibly hard to do with cartesian coordinates, we first convert over to fractional
		self.pos_data = self.get_dir_positions()

		# Make sure that we have a copy of the coordinates that we can work off of
		original_positions = deepcopy(self.pos_data)
		original_atom_list = deepcopy(self.atom_list)
		original_fixation_data = deepcopy(self.fixation_data)

		# Now we need to loop through all of the replications that we need to make
		# Keep in mind that a rep=1 means that no replication in that direction happens (think of them as multipliers almost)
		for xnum in range(xrep):
			for ynum in range(yrep):
				for znum in range(zrep):
					
					# Make sure that we're not going to replicate the positions that alreadu exist
					if not(xnum + ynum + znum == 0):
						replicated_positions = deepcopy(original_positions)
						
						# Add the replicated positions
						replicated_positions[:,0] += xnum
						replicated_positions[:,1] += ynum
						replicated_positions[:,2] += znum
						
						# Append the atoms
						self.pos_data = np.append(self.pos_data,replicated_positions,axis=0)
						self.atom_list = np.append(self.atom_list,original_atom_list)

						# If we have selective dynamics, make sure to add the flags alongside the position data
						if self.selective_dynamics:
							self.fixation_data = np.append(self.fixation_data,original_fixation_data,axis=0)
		
		# Increase the lengths of the lattice vectors accordingly
		self.factored_lattice_parameters[0] = self.factored_lattice_parameters[0] * xrep
		self.factored_lattice_parameters[1] = self.factored_lattice_parameters[1] * yrep
		self.factored_lattice_parameters[2] = self.factored_lattice_parameters[2] * zrep

		# Make sure that the un-factored lattice parameters are updated as well
		self.lattice_parameters = self.lattice_scalar * self.factored_lattice_parameters

		# Recalculate the direct coordinates to be between 0 and 1 (we already adjusted the lattice vectors)
		self.pos_data[:,0] = self.pos_data[:,0] / xrep
		self.pos_data[:,1] = self.pos_data[:,1] / yrep
		self.pos_data[:,2] = self.pos_data[:,2] / zrep

		# If we originally had cartesian coordinates, revert them back to them
		if self.coordinate_type == 'cart':
			self.pos_data = self.get_cart_positions()

		# Remake the atom header using all of the new atoms
		self.remake_atom_header()

	# Create the ICORE POSCAR for the atom in index atomnum
	def create_XPS_POSCAR(self,atomnum):
		if self.selective_dynamics:
			atom_selective = self.fixation_data[atomnum]
			self.fixation_data = np.delete(self.fixation_data,atomnum,0)
			self.fixation_data = np.append(self.fixation_data,[atom_selective],axis=0)
			
			
		atomtype = self.atom_list[atomnum]
		atompos = self.pos_data[atomnum]
		
		self.pos_data = np.delete(self.pos_data,atomnum,0)
		self.atom_list = np.delete(self.atom_list,atomnum,0).tolist()
		
		self.remake_atom_header()
		
		self.atom_list = np.append(self.atom_list,atomtype)
		self.pos_data = np.append(self.pos_data,[atompos],axis=0)
		
		self.atom_header = np.append(np.array(self.atom_header),[[atomtype],[1]],axis=1).tolist()

	# Return a list of indicies of the atoms of 'atomtype' in the atom_list
	def get_atom_type_match_indicies(self,atomtype):
		return np.argwhere(self.atom_list == atomtype).T[0].tolist()

	# Write out information with the atom_order style atomic numbering for lammps
	def write_out_lammps(self,filepath,atom_order):
		if os.path.isfile(filepath):
			os.remove(filepath)
		
		a = np.linalg.norm(self.lattice_parameters[0])
		b = np.linalg.norm(self.lattice_parameters[1])
		c = np.linalg.norm(self.lattice_parameters[2])

		# Lattice angles in radians
		alpha = np.arccos(np.dot(self.lattice_parameters[1],self.lattice_parameters[2])/(b*c))
		beta  = np.arccos(np.dot(self.lattice_parameters[0],self.lattice_parameters[2])/(a*c))
		gamma = np.arccos(np.dot(self.lattice_parameters[0],self.lattice_parameters[1])/(a*b))

		cart_pos = self.get_cart_positions()

		lx = a
		xy = b*np.cos(gamma)
		xz = c*np.cos(beta)
		ly = np.sqrt(b**2-xy**2)
		yz = (b*c*np.cos(alpha)-xy*xz)/ly
		lz = np.sqrt(c**2-xz**2-yz**2)

		with open(filepath,'a') as f:
			# Write out the lattice information
			f.write(f'\n{len(self.atom_list)} atoms\n{len(atom_order)} atom types\n   0.00000   {lx:>10.5f} xlo xhi\n   0.00000   {ly:>10.5f} ylo yhi\n   0.00000   {lz:>10.5f} zlo zhi\n{xy:>10.5f}   {xz:>10.5f}   {yz:>10.5f} xy xz yz\n\nMasses\n\n')
			
			# Get the dictionary of mass data
			mass_dict = getMassDict()
			
			for atomnum,atom in enumerate(atom_order):
				# Calcualte the mass of the atom in amu
				atom_mass = mass_dict[atom]*6.0221409E23*1000

				# Write the atom mass
				f.write(f'{atomnum+1:>6} {atom_mass:>10.5f}\n')
			
			f.write(f'\nAtoms\n\n')

			for atomnum,atom in enumerate(self.atom_list):
				atom_car_pos = cart_pos[atomnum]
				f.write(f'  {atomnum+1:>4}  {atom_order.index(atom)+1:>4} {atom_car_pos[0]:>15.10f} {atom_car_pos[1]:>15.10f} {atom_car_pos[2]:>15.10f}\n')

	def write_out_potcar(self,filepath,potcar_root):
		
		# Make sure that potcar_root ends with / (that way it's a directory)
		if potcar_root[-1] != '/':
			potcar_root += '/'

		potcar = ''
		for atom in self.atom_header[0]:
			with open(potcar_root + atom + '/POTCAR','r') as f:
				potcar += f.read()

		with open(filepath,'a') as f:
			f.write(potcar)

	# Write out the POSCAR file
	def write_out(self,filepath):
		if os.path.isfile(filepath):
			os.remove(filepath)
		
		with open(filepath,'a') as f:
			f.write(self.name + '\n')
			f.write(str(self.lattice_scalar) + '\n')
			for row in self.factored_lattice_parameters:
				f.write('   '.join([f'{i:15.10f}' for i in row]) + '\n')
			
			for row in self.atom_header:
				f.write('  '.join([f'{i:>4}' for i in row]) + '\n')
			
			if self.selective_dynamics:
				f.write('Selective Dynamics\n')
			
			if self.coordinate_type == 'dir':
				f.write('Direct\n')
			else:
				f.write('Cartesian\n')
			
			for rownum,row in enumerate(self.pos_data):
				f.write('  '.join([f'{i:20.13f}' for i in row]))
				if self.selective_dynamics:
					f.write('   ' + ' '.join(self.fixation_data[rownum]) + '\n')
				else:
					f.write('\n')
				

class INCAR:
	def __init__(self,filepath):
		
		with open(filepath,'r') as f:
			self.file_data = [i.split() for i in f.read().splitlines()]
		
		self.tags = {}
		for fields in self.file_data:
			if len(fields) != 0:
				if len(" ".join(fields).split("!")[0]) != 0:
					line = " ".join(fields).split("!")[0]
					tagname = line.split("=")[0].split()[0]
					tagdat = line.split("=")[1].split()
					self.tags[tagname] = tagdat

	def get_sorted_tags_string(self):
		taglist = '\n'.join([str(tag) + ' = ' + ' '.join([str(i) for i in self.tags[tag]]) for tag in self.tags])
		return taglist
	
	def write_out(self,filepath):
		if os.path.isfile(filepath):
			os.remove(filepath)
		
		with open(filepath,'a') as f:
			f.write(self.get_sorted_tags_string())		

# Make XDATCARs using heading information from POSCARs and a list of positions for each frame
# Can also use POSCAR data to fill out the lattice info and headers
class XDATCAR:
	def __init__(self,filepath,*,is_str_not_file=False):
		
		if is_str_not_file:
			self.file_data = [i.split() for i in filepath.splitlines()]
		else:
			with open(filepath,'r') as f:
				self.file_data = [i.split() for i in f.read().splitlines()]

		self.name = ' '.join(self.file_data[0])

		# Get the lattice parameter data
		self.lattice_scalar = float(self.file_data[1][0])
		self.factored_lattice_parameters = np.array(self.file_data[2:5]).astype(np.float64)
		self.lattice_parameters = self.lattice_scalar * self.factored_lattice_parameters
		self.volume = np.linalg.det(self.lattice_parameters)

		# Get atom header data
		self.atom_header = [self.file_data[5],[int(i) for i in self.file_data[6]]]
		self.atom_list = []
		for speciesnum,species in enumerate(self.atom_header[0]):
			for occur in range(self.atom_header[1][speciesnum]):
				self.atom_list.append(species)

		self.atom_list = np.array(self.atom_list)

		# Get atomic positions by finding lines with 'configuration' in them (must be Direct, think this is normal anyhow)
		self.pos_data = []
		for num,line in enumerate(self.file_data):
			if 'configuration' in ' '.join(line):
				self.pos_data.append(self.file_data[num+1:num+len(self.atom_list)+1])
		
		self.pos_data = np.array(self.pos_data).astype(float)

	# Loads the positions of a POSCAR into the XDATCAR (needs same lattice and number of atoms)
	def load_poscar_frame(self,poscar_object):
		if not len(self.pos_data) == 0:
			self.pos_data = np.append(self.pos_data,np.array([poscar_object.get_dir_positions()]),axis=0)
		else:
			self.pos_data = np.array([poscar_object.get_dir_positions()])

	# Scan through all frames of an atom looking for movements across boundaries and change the atom position if they exist
	# A direct coordinate movement threshold of 0.75 is set as a default
	def handle_periodicity(self,*,threshold=0.75):
		# For each atom in all frames (except the first one) and in all three directions
		for atomnum in range(len(self.atom_list)):
			for frame_num,frame in enumerate(self.pos_data):
				if not frame_num == 0:
					for direction_num,direction in enumerate(frame[atomnum]):
						# Calculate the movement of the atom in a direction between steps
						displacement = direction - self.pos_data[frame_num-1][atomnum][direction_num]
						
						# If the atom moved across the negative boundary
						if displacement > threshold:
							self.pos_data[frame_num][atomnum][direction_num] -= 1
						
						# If the atom moved across the positive boundary
						elif displacement < -threshold:
							self.pos_data[frame_num][atomnum][direction_num] += 1

	# Perform cubic interpolation of XDATCAR frames with a specified number of frames between.
	# Hermite smoothing means that the atoms smoothly stop at the original XDATCAR frames before moving on.
	# Hermite fixed frames is a list of boolean values for each frame in order on whether its derivatives are set to zero.
	# Note: This operation requires boundary-periodic interactions to be accounted for by creating direct coordinates less than 0 or greater than 1.
	def smooth_trajectories(self,frames_between,*,hermite_smooth=False,hermite_fixed_frames=[],periodicity_threshold=0.75):
		self.handle_periodicity(threshold=periodicity_threshold)

		# Convert all of the coordinates to Cartesian
		self.pos_data = np.array([np.matmul((self.lattice_scalar*self.factored_lattice_parameters),frame.T).T for frame in self.pos_data])

		# Rearrange the positions so that its atom -> frame -> position
		self.pos_data = np.array([self.pos_data[:,atom_num] for atom_num,atom in enumerate(self.atom_list)])
		
		def get_3D_spline(tdata,xyzdata):
			
			xspline = si.CubicSpline(x=tdata,y=xyzdata[:,0])
			yspline = si.CubicSpline(x=tdata,y=xyzdata[:,1])
			zspline = si.CubicSpline(x=tdata,y=xyzdata[:,2])
			
			if hermite_smooth:
				# We want the velocities of all ions to be zero at the POSCARs
				dudt = np.array([xspline(tdata,1),yspline(tdata,1),zspline(tdata,1)]).T

				dudt[np.array(hermite_fixed_frames),:] = 0

				xspline = si.CubicHermiteSpline(x=tdata,y=xyzdata[:,0],dydx=dudt[:,0])
				yspline = si.CubicHermiteSpline(x=tdata,y=xyzdata[:,1],dydx=dudt[:,1])
				zspline = si.CubicHermiteSpline(x=tdata,y=xyzdata[:,2],dydx=dudt[:,2])
				return lambda t: np.array([xspline(t),yspline(t),zspline(t)])
			
			else:
				return lambda t: np.array([xspline(t),yspline(t),zspline(t)])

		frame_function = lambda t: np.array([get_3D_spline(tdata=np.linspace(0,1,endpoint=True,num=len(self.pos_data[0])),xyzdata=atom_pos_data)(t) for atom_pos_data in self.pos_data])
		number_of_frames = len(self.pos_data[0] - 1) * (frames_between + 1) + 1
		sample_time = np.linspace(start=0,stop=1,endpoint=True,num=number_of_frames)

		self.pos_data = np.array([frame_function(t) for t in sample_time])
		self.pos_data = np.array([np.matmul(np.linalg.inv(self.lattice_scalar*self.factored_lattice_parameters),frame.T).T for frame in self.pos_data])
	
	# Write out the XDATCAR file
	def write_out(self,filepath):
		if os.path.isfile(filepath):
			os.remove(filepath)
		
		with open(filepath,'a') as f:
			f.write(self.name + '\n')
			f.write(str(self.lattice_scalar) + '\n')
			for row in self.factored_lattice_parameters:
				f.write('  ' + ' '.join([f'{i:11.6f}' for i in row]) + '\n')
			
			for row in self.atom_header:
				f.write(' ' + '  '.join([f'{i:>4}' for i in row]) + '\n')
			
			for frame_num,frame in enumerate(self.pos_data):
				f.write(f'Direct configuration={str(frame_num+1):>6}\n')
				for rownum,row in enumerate(frame):
					f.write('  ' + ' '.join([f'{i:>11.8f}' for i in row]))
					f.write('\n')
	


class OUTCAR:
	def __init__(self,filepath):
		# Set some constants that will be checked against in the file read
		self.is_converged = False
		self.volume = None
		self.ionic_steps = 0
		self.selective_dynamics = False
		self.elapsed_time = None
		self.e_sigzero = []
		self.forces = []
		self.mag_forces = []
		self.is_neb = False
		self.tangent_vectors = []

		# By only reading the OUTCARs once, the operation time is cut significantly (althought the string searching takes a while)
		with open(filepath,'r') as f:
			self.file_data = f.read().splitlines()
			for num,line in enumerate(self.file_data):
				# This indicates a converged file
				if 'reached required accuracy - stopping structural energy minimisation' in line:
					self.is_converged = True
				
				# Grab the volume of the cell
				elif 'volume of cell' in line:
					self.volume = float(line.split()[4])
				
				# This only appears at the end of an ionic step, also shows the energy after the step
				elif 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM' in line:
					self.ionic_steps += 1
					self.e_sigzero.append(float(self.file_data[num+4].split('=')[-1].split()[0]))
				
				# The time elapsed
				elif 'Elapsed time (sec)' in line:
					self.elapsed_time = float(line.split()[3])

				# The forces listed at the end of each ionic step
				elif 'TOTAL-FORCE' in line:
					forces = []
					for force_line in self.file_data[num+2:]:
						# There is a nice line drawn at the end of the forces, stop looping once you see it
						if '------------' in force_line:
							break
						else:
							forces.append([float(i) for i in force_line.split()[3:]])
					
					# We want the forces of all of the atoms
					self.forces.append(forces)

					# Get the magnitude of all of the force vectors
					self.mag_forces.append([np.linalg.norm(i) for i in self.forces[-1]])

				# Check if the OUTCAR is for an NEB
				elif 'CHAIN: Running the NEB' in line:
					self.is_neb = True
				
				# Grab NEB tangent band direction unit vectors
				elif 'NEB: Tangent' in line:
					tangent_vectors = []
					for tangent_vector_line in self.file_data[num+2:]:
						# There is a blank line at the end of the tangent section, stop looking once you see it
						if len(tangent_vector_line.split()) == 0:
							break
						else:
							# Turn the vector elements into a float
							vector = np.array([float(i) for i in tangent_vector_line.split()])

							# Turn the vector into an actual unit vector by dividing the original magnitude
							if not np.linalg.norm(vector) == 0:
								vector = vector / np.linalg.norm(vector)
							else:
								vector = np.array([0,0,0])
							
							# The way that numpy handles appending confuses me - I like doing it all in python lists
							tangent_vectors.append(vector.tolist())

					# We want the vectors of all of the atoms
					self.tangent_vectors.append(tangent_vectors)

			# If forces were found, turn them into arrays
			if len(self.forces) > 0:
				self.forces = np.array(self.forces)
				self.mag_forces = np.array(self.mag_forces)

			# If NEB tangent vectors were found, turn them into arrays
			if len(self.tangent_vectors) > 0:
				self.tangent_vectors = np.array(self.tangent_vectors)

# Return a dictionary of atomic masses in kg
def getMassDict():
	mass = {
	"H": 1.67377E-27,
	"He": 6.64648E-27,
	"Li": 1.15258E-26,
	"Be": 1.49651E-26,
	"B": 1.79504E-26,
	"C": 1.99447E-26,
	"N": 2.32587E-26,
	"O": 2.65676E-26,
	"F": 3.15476E-26,
	"Ne": 3.3508E-26,
	"Na": 3.81754E-26,
	"Mg": 4.03594E-26,
	"Al": 4.48039E-26,
	"Si": 4.66371E-26,
	"P": 5.14332E-26,
	"S": 5.32369E-26,
	"Cl": 5.88711E-26,
	"K": 6.49243E-26,
	"Ar": 6.63353E-26,
	"Ca": 6.65544E-26,
	"Sc": 7.46511E-26,
	"Ti": 7.95399E-26,
	"V": 8.45904E-26,
	"Cr": 8.63414E-26,
	"Mn": 9.12267E-26,
	"Fe": 9.27362E-26,
	"Ni": 9.74737E-26,
	"Co": 9.78609E-26,
	"Cu": 1.05521E-25,
	"Zn": 1.08566E-25,
	"Ga": 1.15773E-25,
	"Ge": 1.20539E-25,
	"As": 1.2441E-25,
	"Se": 1.31116E-25,
	"Br": 1.32684E-25,
	"Kr": 1.39153E-25,
	"Rb": 1.41923E-25,
	"Sr": 1.45497E-25,
	"Y": 1.47632E-25,
	"Zr": 1.51474E-25,
	"Nb": 1.54275E-25,
	"Mo": 1.59312E-25,
	"Tc": 1.62733E-25,
	"Ru": 1.67831E-25,
	"Rh": 1.70879E-25,
	"Pd": 1.76681E-25,
	"Ag": 1.79119E-25,
	"Cd": 1.86661E-25,
	"In": 1.90663E-25,
	"Sn": 1.97089E-25,
	"Sb": 2.02171E-25,
	"I": 2.1073E-25,
	"Te": 2.11885E-25,
	"Xe": 2.18029E-25,
	"Cs": 2.20695E-25,
	"Ba": 2.28042E-25,
	"La": 2.30658E-25,
	"Ce": 2.32675E-25,
	"Pr": 2.33983E-25,
	"Nd": 2.39516E-25,
	"Pm": 2.40778E-25,
	"Sm": 2.49745E-25,
	"Eu": 2.52336E-25,
	"Gd": 2.6112E-25,
	"Tb": 2.63902E-25,
	"Dy": 2.69838E-25,
	"Ho": 2.73874E-25,
	"Er": 2.77742E-25,
	"Tm": 2.80522E-25,
	"Yb": 2.8734E-25,
	"Lu": 2.9054E-25,
	"Hf": 2.9639E-25,
	"Ta": 3.00471E-25,
	"W": 3.0529E-25,
	"Re": 3.09204E-25,
	"Os": 3.15835E-25,
	"Ir": 3.19189E-25,
	"Pt": 3.23955E-25,
	"Au": 3.27071E-25,
	"Hg": 3.33088E-25,
	"Tl": 3.39365E-25,
	"Pb": 3.44064E-25,
	"Bi": 3.4702E-25,
	"Po": 3.47053E-25,
	"At": 3.48713E-25,
	"Rn": 3.6864E-25,
	"Fr": 3.703E-25,
	"Ra": 3.75324E-25,
	"Ac": 3.76989E-25,
	"Pa": 3.83644E-25,
	"Th": 3.85309E-25,
	"Np": 3.93628E-25,
	"U": 3.95257E-25,
	"Pu": 4.01851E-25,
	"Am": 4.03511E-25,
	"Bk": 4.10153E-25,
	"Cm": 4.10153E-25,
	"No": 4.15135E-25,
	"Cf": 4.16796E-25,
	"Es": 4.18456E-25,
	"Hs": 4.23438E-25,
	"Mt": 4.25098E-25,
	"Fm": 4.26759E-25,
	"Md": 4.28419E-25,
	"Lr": 4.3174E-25,
	"Rf": 4.33401E-25,
	"Bh": 4.35061E-25,
	"Db": 4.35061E-25,
	"Sg": 4.36722E-25,
	"Uun": 4.46685E-25,
	"Uuu": 4.51667E-25,
	"Uub": 4.5997E-25
	}
	return mass
