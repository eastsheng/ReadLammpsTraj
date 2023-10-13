# calculating the density,msd and rdf from lammps traj
# id mol type mass x y z vx vy vz fx fy fz q
import numpy as np 
import pandas as pd
from tqdm import tqdm
import datetime
from itertools import islice

def __version__():
	version = "1.1.9"
	return version

def __print_version__():
    cloud = [
			"______                   _    _       _____                 _ ",
			"| ___ \                 | |  | |     |_   _|               (_)",
			"| |_/ /  ___   __ _   __| |  | |       | |   _ __   __ _    _ ",
			"|    /  / _ \ / _` | / _` |  | |       | |  | '__| / _` |  | |",
			"| |\ \ |  __/| (_| || (_| |  | |____   | |  | |   | (_| |  | |",
			"\_| \_| \___| \__,_| \__,_|  \_____/   \_/  |_|    \__,_|  | |",
			"                                                          _/ |",
			"                                                         |__/ ",
    ]
    n = 32
    print(n*"- ")
    print(n*". ")
    for line in cloud:
        print(line)
    version = __version__()
    print('@ReadLammpsTraj-'+version,", Good Luck!")
    print(n*". ")
    print(n*"- ")
    current_datetime = datetime.datetime.now()
    return print("Time:",current_datetime)



def read_mass(lammpsdata):
	"""
	read atomic mass from lammps data. 
	"""
	data = open(lammpsdata,"r")
	lines = data.read()
	mass = []
	(header,mass_other)=lines.split("\n\nMasses\n\n")
	try:
		try:
			try:
				(mass,other)=mass_other.split("\n\nPair Coeffs ")
			except:
				(mass,other)=mass_other.split("\n\nBond Coeffs ")
		except:
			(mass,other)=mass_other.split("\n\nAngle Coeffs ")
	except:
		(mass,other)=mass_other.split("\n\nAtoms # ")

	mass=list(mass.split('\n'))
	mass = list(filter(None, mass))
	mass_id,mass_sub = [], []
	for i in range(len(mass)):
		mass_uu = mass[i].strip(" ").split(" ")
		mass_id.append(int(mass_uu[0]))
		mass_sub.append(float(mass_uu[1]))

	pairs = zip(mass_id,mass_sub)
	mass_dict = {k: v for k, v in pairs}

	return mass_dict



class ReadLammpsTraj(object):
	"""docstring for ClassName"""
	def __init__(self,f,timestep=1):
		super(ReadLammpsTraj, self).__init__()
		self.f = f
		self.amu2g = 6.02214076208112e23
		self.A2CM = 1e-8 
		self.timestep=timestep#fs

	def read_info(self,):
		with open(self.f,'r') as f:
			L1 = f.readline()
			L2 = f.readline()
			L3 = f.readline()
			L4 = f.readline()
			L5 = f.readline()
			L6 = f.readline()
			L7 = f.readline()
			L8 = f.readline()
			L9 = f.readline().strip().split()[2:]#列标签
			self.col = L9
			step1 = int(L2)
			self.atom_n = int(L4)
			self.xlo,self.xhi = float(L6.split()[0]),float(L6.split()[1])
			self.ylo,self.yhi = float(L7.split()[0]),float(L7.split()[1])
			self.zlo,self.zhi = float(L8.split()[0]),float(L8.split()[1])
			self.Lx = self.xhi-self.xlo
			self.Ly = self.yhi-self.ylo
			self.Lz = self.zhi-self.zlo
			self.vlo = self.Lx*self.Ly*self.Lz
			for i in range(self.atom_n+1):
				Li = f.readline()
				# print(Li)
			try:
				step2 = int(f.readline())
				self.step_inter = step2-step1
				# print("Step interval:",self.step_inter,"\nAtom number:",self.atom_n)
				# print("xlo:",self.xlo,"xhi:",self.xhi,"Lx:",self.Lx)
				# print("ylo:",self.ylo,"yhi:",self.yhi,"Ly:",self.Ly)
				# print("zlo:",self.zlo,"zhi:",self.zhi,"Lz:",self.Lz)
			except:
				self.step_inter = 0
				# print("pass")
		return self.step_inter,self.atom_n,self.Lx,self.Ly,self.Lz

	def read_header(self,nframe):
		# print("--- Start read header of %s th frame ---" %nframe)

		skip = int(9*(nframe)+self.atom_n*(nframe))
		header = []
		
		with open(self.f,'r') as f:
			for line in islice(f,skip,skip+9):
				header.append(line)

		# with open(self.f,'r') as f:
		# 	for n in range(skip):
		# 		f.readline()
		# 	for i in range(9):
		# 		line = f.readline()
		# 		header.append(line)
		
		print("--- Read header of %s th frame done! ---" %nframe)
		return header

	def read_vol(self,nframe):
		"""
		calculate the volume by traj file
		nframe: n_th frame, nframe>=1 and int type
		vol: volume of system, unit: A^3
		"""
		skip = 5*(nframe+1)+(self.atom_n+4)*(nframe)
		vol_xyz = np.loadtxt(self.f,skiprows=skip,max_rows=3)
		# print(vol_xyz)
		xL = vol_xyz[0,1]-vol_xyz[0,0]
		yL = vol_xyz[1,1]-vol_xyz[1,0]
		zL = vol_xyz[2,1]-vol_xyz[2,0]
		vol = xL*yL*zL
		return  vol

	def read_mxyz(self,nframe):
		"""
		read mass, and x, y, z coordinates of nth frame from traj...
		nframe: number of frame 
		"""
		traj = self.read_traj(nframe)
		try:
			self.mol = traj.loc[:,"mol"].values.astype(np.int64)#id mol type
		except:
			print("No molecule types in traj...")

		try:
			self.atom = traj.loc[:,"type"].values.astype(np.int64)#id atom type
		except:
			print("No atom types in traj...")

		xyz = traj.loc[:,"x":"z"].values.astype(np.float64) # x y z

		try:
			mass = traj.loc[:,"mass"].values.astype(np.float64)#mass
		except:
			print("No mass out in traj...")
			mass = np.zeros(len(xyz))
		mxyz = np.hstack((mass.reshape(-1,1),xyz))

		position = mxyz

		return position



	def read_mxyz_add_mass(self,nframe,atomtype_list,mass_list):
		# 不区分分子类型，计算所有的密度所需
		traj = self.read_traj(nframe)
		# print(traj)
		self.mol = traj.loc[:,"mol"].values.astype(np.int64)#id mol type
		self.atom = traj.loc[:,"type"].values.astype(np.int64)#id atom type
		xyz = traj.loc[:,"x":"z"].values.astype(np.float64) # x y z
		# print(xyz.shape)
		mass_array = np.zeros(len(xyz)).reshape(-1,1)
		for i in range(len(xyz)):
			for j in range(len(mass_list)):
				if self.atom[i] == atomtype_list[j]:
					mass_array[i] = mass_list[j]
		mxyz = np.hstack((mass_array,xyz))
		return mxyz


	def read_traj(self,nframe):
		"""
		read data of nth frame from traj...
		nframe: number of frame 
		"""
		skip = 9*(nframe+1)+self.atom_n*(nframe)
		traj = np.loadtxt(self.f,skiprows=skip,max_rows=self.atom_n,dtype="str")
		# print("Labels in traj is:",self.col)
		traj = pd.DataFrame(traj,columns=self.col)
		# print(traj)
		return traj

	def oneframe_alldensity(self,mxyz,Nbin,mass_dict={},density_type="mass",direction="z"):
		"""
		calculating density of all atoms......
		mxyz: array of mass, x, y, and z;
		Nbin: number of bins in x/y/z-axis
		mass_dict: masses of atoms ,default={} 
		density_type: calculated type of density 
		"""

		unitconvert = self.amu2g*(self.A2CM)**3
		if direction=="z" or direction=="Z":
			dr = self.Lz/Nbin #z方向bin
			L = mxyz[:,3]
			lo = self.zlo
			vlo = (self.Lx*self.Ly*dr)*unitconvert
		elif direction=="y" or direction=="Y":
			dr = self.Ly/Nbin
			L = mxyz[:,2]
			lo = self.ylo
			vlo = (self.Lx*self.Lz*dr)*unitconvert
		elif direction=="x" or direction=="X":
			dr = self.Lx/Nbin
			L = mxyz[:,1]
			lo = self.xlo
			vlo = (self.Ly*self.Lz*dr)*unitconvert

		mass_key=list(mass_dict.keys())
		for i in range(len(self.atom)):
			for j in range(len(mass_key)):
				if self.atom[i] == mass_key[j]:
					mxyz[i,0] = mass_dict[mass_key[j]]
		MW = mxyz[:,0] #相对分子质量

		if np.all(MW==0):
			density_type = "number"
			print("\nNo provided mass, will calculate number density!\n")
		else:
			density_type = "mass"

		rho_n = [] #average density list in every bins
		lc_n  = []
		for n in range(Nbin):
			mass_n=0 #tot mass in bin
			l0 = lo+dr*n #down coord of bin
			l1 = lo+dr*(n+1)#up coord of bin
			lc = (l0+l1)*0.5
			for i in range(self.atom_n):
				if L[i]>=l0 and L[i]<=l1:
					if density_type == "mass":
						mass_n = MW[i]+mass_n
					else:
						mass_n = mass_n+1
			rho = mass_n/vlo
			# print(rho)
			rho_n.append(rho)
			lc_n.append(lc)
		lc_n = np.array(lc_n).reshape(-1,1)
		rho_n = np.array(rho_n).reshape(-1,1)

		return lc_n,rho_n

	def oneframe_moldensity(self,mxyz,Nbin,id_range,mass_dict={},id_type="mol",density_type="mass",direction="z"):
		"""
		calculating density of some molecules......
		mxyz: array of mass, x, y, and z;
		Nbin: number of bins in x/y/z-axis
		id_range: range of molecule/atom id;
		mass_dict: masses of atoms ,default={} 
		id_type: according to the molecule/atom id, to recognize atoms, args: mol, atom
		density_type: calculated type of density 
		"""
		unitconvert = self.amu2g*(self.A2CM)**3
		if direction=="z" or direction=="Z":
			dr = self.Lz/Nbin #z方向bin
			L = mxyz[:,3]
			lo = self.zlo
			vlo = (self.Lx*self.Ly*dr)*unitconvert
		elif direction=="y" or direction=="Y":
			dr = self.Ly/Nbin
			L = mxyz[:,2]
			lo = self.ylo
			vlo = (self.Lx*self.Lz*dr)*unitconvert
		elif direction=="x" or direction=="X":
			dr = self.Lx/Nbin
			L = mxyz[:,1]
			lo = self.xlo
			vlo = (self.Ly*self.Lz*dr)*unitconvert

		mass_key=list(mass_dict.keys())
		for i in range(len(self.atom)):
			for j in range(len(mass_key)):
				if self.atom[i] == mass_key[j]:
					mxyz[i,0] = mass_dict[mass_key[j]]
		MW = mxyz[:,0] #相对分子质量
		if np.all(MW==0):
			density_type = "number"
			print("\nNo provided mass, will calculate number density!\n")
		else:
			density_type = "mass"

		rho_n = [] #average density list in every bins
		lc_n  = []
		# print(MW.shape,Z.shape)
		if id_type == "mol":
			id_know = self.mol
		elif id_type == "atom":
			id_know = self.atom
		for n in range(Nbin):
			mass_n=0 #tot mass in bin
			l0 = lo+dr*n #down coord of bin
			l1 = lo+dr*(n+1)#up coord of bin
			lc = (l0+l1)*0.5
			# print(z0,z1,zc)
			for i in range(self.atom_n):
				if id_know[i]>=id_range[0] and id_know[i]<=id_range[1]:
					# if i atom in [z0:z1]
					if L[i]>=l0 and L[i]<=l1:
						if density_type == "mass":
							mass_n = MW[i]+mass_n
						else:
							mass_n = mass_n+1
			rho = mass_n/vlo
			# print(rho)
			rho_n.append(rho)
			lc_n.append(lc)
		lc_n = np.array(lc_n).reshape(-1,1)
		rho_n = np.array(rho_n).reshape(-1,1)	
		return lc_n,rho_n

	def TwoD_Density(self,mxyz,atomtype_n,Nx=1,Ny=1,Nz=1,mass_or_number="mass",id_type="mol"):
		'''
		mxyz: mass x y z
		atom_n: tot number of atoms
		atomtype_n: type of molecules,list,atom_n=[1,36], the 1 is the first atom type and 36 is the last one atom type
		Nx,Ny,Nz: layer number of x , y, z for calculating density, which is relate to the precision of density,
		and default is 1, that is, the total density.
		mass_or_number: "mass: mass density; number: number density"
		id_type:"mol" or "atom" for atomtype_n
		'''
		unitconvert = self.amu2g*(self.A2CM)**3
		dX = self.Lx/Nx #x方向bin
		dY = self.Ly/Ny #y方向bin
		dZ = self.Lz/Nz #z方向bin
		MW = mxyz[:,0] #相对分子质量
		X = mxyz[:,1] #x
		Y = mxyz[:,2] #y
		Z = mxyz[:,3] #z
		if id_type == "mol":
			id_know = self.mol
		elif id_type == "atom":
			id_know = self.atom
		xc_n,yc_n,zc_n = [],[],[]
		rho_n = [] #average density list in every bins
		for xi in tqdm(range(Nx)):
			x0 = self.xlo+dX*xi #down coord of bin
			x1 = self.xlo+dX*(xi+1) #down coord of bin
			xc = (x0+x1)*0.5
			xc_n.append(xc)
			# print(xi,'---Nx:---',Nx)
			for yi in range(Ny):
				# print(yi,'---Ny:---',Ny)
				y0 = self.ylo+dY*yi #down coord of bin
				y1 = self.ylo+dY*(yi+1) #down coord of bin
				yc = (y0+y1)*0.5
				# print(yc)
				yc_n.append(yc)
				for zi in range(Nz):
					# print(zi,'---Nz:---',Nz)
					z0 = self.zlo+dZ*zi #down coord of bin
					z1 = self.zlo+dZ*(zi+1) #down coord of bin
					zc = (z0+z1)*0.5
					zc_n.append(zc)
		
					n=0 #tot mass or number in bin

					for i in range(self.atom_n):
						
						if id_know[i]>=atomtype_n[0] and id_know[i]<=atomtype_n[1]:
							if X[i]>=x0 and X[i]<=x1 and Y[i]>=y0 and Y[i]<=y1 and Z[i]>=z0 and Z[i]<=z1:
								if mass_or_number == "mass":
									n = MW[i]+n
								elif mass_or_number == "number":
									n = n+1
									# print(i,'---',self.atom_n,MW[i])
					vlo = (dX*dY*dZ)*unitconvert
					rho = n/vlo
					rho_n.append(rho)

		xc_n = np.array(xc_n)
		xc_n = np.unique(xc_n).reshape((Nx,1))

		yc_n = np.array(yc_n)
		yc_n = np.unique(yc_n).reshape((Ny,1))

		zc_n = np.array(zc_n)
		zc_n = np.unique(zc_n).reshape((Nz,1))
		rho_nxyz = np.array(rho_n).reshape((Nx,Ny,Nz))

		minx = min(xc_n)
		miny = min(yc_n)
		minz = min(zc_n)
		xc_n = xc_n-minx
		yc_n = yc_n-miny
		zc_n = zc_n-minz
		
		# print(xc_n,yc_n,zc_n,rho_nxyz)

		return xc_n,yc_n,zc_n,rho_nxyz

	def unwrap(self,dn,dm,Lr):
		dr = dn-dm
		if abs(dr) > 0.5*Lr:
			dr = dr - Lr*(np.sign(dr))
		else:
			dr = dr
		return dr
		
	def zoning(self,sorted_traj,axis_range,direc="y"):
		"""
		Divide a coordinate interval along a direction, such as, x or y or z
		sorted_traj: sorted lammps traj, pandas dataframe format, it includes at least 'id mol type x y z'
		axis_range: Divide interval, a list, such as, axis_range = [0,3.5], unit/Angstrom
		direc: The direction to be divided, default direc="y"
		"""
		m,n = sorted_traj.shape
		if direc=="X":
			direc = "x"
		elif direc=="Y":
			direc = "y"
		elif direc=="Z":
			direc = "z"
		else:
			direc = "y"
		# whether in the interval
		condition1 = (sorted_traj[direc].between(axis_range[0],axis_range[1]))
		sorted_zoning_traj = sorted_traj[condition1]
		# Whether it's the same molecule
		mols = sorted_zoning_traj["mol"]
		condition2 = (sorted_traj["mol"].isin(mols))
		sorted_zoning_traj = sorted_traj[condition2]

		return sorted_zoning_traj