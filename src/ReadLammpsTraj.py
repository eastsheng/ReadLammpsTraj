# calculating the density,msd and rdf from lammps traj
# id mol type mass x y z vx vy vz fx fy fz q
import numpy as np 
import pandas as pd

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
				print("Step interval:",self.step_inter,"\nAtom number:",self.atom_n)
				print("xlo:",self.xlo,"xhi:",self.xhi,"Lx:",self.Lx)
				print("ylo:",self.ylo,"yhi:",self.yhi,"Ly:",self.Ly)
				print("zlo:",self.zlo,"zhi:",self.zhi,"Lz:",self.Lz)
			except:
				self.step_inter = 0
				print("pass")
		return self.step_inter,self.atom_n,self.Lx,self.Ly,self.Lz

	def read_header(self,nframe):
		with open(self.f,'r') as f:
			header = []
			for index, line in enumerate(f,1):
				if index >=1+(self.atom_n+9)*(nframe) and index<=9+(self.atom_n+9)*(nframe):
					# print(line)
					header.append(line)
			header = header
			# print(header)
		return header

	def read_vol(self,nframe):
		"""
		calculate the volume by traj file
		nframe: n_th frame, nframe>=1 and int type
		vol: volume of system, unit: A^3
		"""
		skip = 5*nframe+(self.atom_n+4)*(nframe-1)
		vol_xyz = np.loadtxt(self.f,skiprows=skip,max_rows=3)
		# print(vol_xyz)
		xL = vol_xyz[0,1]-vol_xyz[0,0]
		yL = vol_xyz[1,1]-vol_xyz[1,0]
		zL = vol_xyz[2,1]-vol_xyz[2,0]
		vol = xL*yL*zL
		return  vol

	def read_mxyz(self,nframe):
		# 不区分分子类型，计算所有的密度所需
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

	def read_traj(self,nframe):
		# 读取所有数据
		skip = 9*nframe+self.atom_n*(nframe-1)
		traj = np.loadtxt(self.f,skiprows=skip,max_rows=self.atom_n,dtype="str")
		print("Labels in traj is:",self.col)
		traj = pd.DataFrame(traj,columns=self.col)
		print(traj)
		return traj

	def oneframe_alldensity(self,mxyz,Nz,density_type="mass"):
		# 只计算一帧的密度
		unitconvert = self.amu2g*(self.A2CM)**3
		dZ = self.Lz/Nz #z方向bin
		MW = mxyz[:,0] #相对分子质量
		if np.all(MW==0):
			density_type = "number"
			print("\nNo provided mass, will calculate number density!\n")
		else:
			density_type = "mass"

		Z = mxyz[:,3] #z
		rho_n = [] #average density list in every bins
		zc_n  = []
		# print(MW.shape,Z.shape)
		for n in range(Nz):
			mass_n=0 #tot mass in bin
			z0 = self.zlo+dZ*n #down coord of bin
			z1 = self.zlo+dZ*(n+1)#up coord of bin
			zc = (z0+z1)*0.5
			# print(z0,z1,zc)
			for i in range(self.atom_n):
				# if i atom in [z0:z1] 
				if Z[i]>=z0 and Z[i]<=z1:
					if density_type == "mass":
						mass_n = MW[i]+mass_n
					else:
						mass_n = mass_n+1
			# print(mass_n)
			vlo = (self.Lx*self.Ly*dZ)*unitconvert
			# print(vlo)
			rho = mass_n/vlo
			# print(rho)
			rho_n.append(rho)
			zc_n.append(zc)
		zc_n = np.array(zc_n).reshape(-1,1)
		rho_n = np.array(rho_n).reshape(-1,1)

		return zc_n,rho_n

	def oneframe_moldensity(self,mxyz,Nz,mol_n,id_type="mol",density_type="mass"):
		# 只计算一帧的密度
		unitconvert = self.amu2g*(self.A2CM)**3
		dZ = self.Lz/Nz #z方向bin
		MW = mxyz[:,0] #相对分子质量
		if np.all(MW==0):
			density_type = "number"
			print("\nNo provided mass, will calculate number density!\n")
		else:
			density_type = "mass"
		Z = mxyz[:,3] #z
		rho_n = [] #average density list in every bins
		zc_n  = []
		# print(MW.shape,Z.shape)
		if id_type == "mol":
			id_know = self.mol
		elif id_type == "atom":
			id_know = self.atom
		for n in range(Nz):
			mass_n=0 #tot mass in bin
			z0 = self.zlo+dZ*n #down coord of bin
			z1 = self.zlo+dZ*(n+1)#up coord of bin
			zc = (z0+z1)*0.5
			# print(z0,z1,zc)
			for i in range(self.atom_n):
				if id_know[i]>=mol_n[0] and id_know[i]<=mol_n[1]:
					# if i atom in [z0:z1]
					if Z[i]>=z0 and Z[i]<=z1:
						if density_type == "mass":
							mass_n = MW[i]+mass_n
						else:
							mass_n = mass_n+1
			# print(mass_n)
			vlo = (self.Lx*self.Ly*dZ)*unitconvert
			# print(vlo)
			rho = mass_n/vlo
			# print(rho)
			rho_n.append(rho)
			zc_n.append(zc)
		zc_n = np.array(zc_n).reshape(-1,1)
		rho_n = np.array(rho_n).reshape(-1,1)	
		return zc_n,rho_n

	def TwoD_Density(self,mxyz,mol_n,Nx=1,Ny=1,Nz=1,mass_or_number="mass"):
		'''
		mxyz: mass x y z
		atom_n: tot number of atoms
		mol_n: type of molecules,list,mol_n=[1,36], the 1 is the first mol type and 36 is the last one mol type
		Nx,Ny,Nz: layer number of x , y, z for calculating density, which is relate to the precision of density,
		and default is 1, that is, the total density.
		mass_or_number: "mass: mass density; number: number density"
		'''
		unitconvert = self.amu2g*(self.A2CM)**3
		dX = self.Lz/Nx #x方向bin
		dY = self.Lz/Ny #y方向bin
		dZ = self.Lz/Nz #z方向bin
		MW = mxyz[:,0] #相对分子质量
		X = mxyz[:,1] #x
		Y = mxyz[:,2] #y
		Z = mxyz[:,3] #z
		xc_n,yc_n,zc_n = [],[],[]
		rho_n = [] #average density list in every bins
		for xi in range(Nx):
			x0 = self.xlo+dX*xi #down coord of bin
			x1 = self.xlo+dX*(xi+1) #down coord of bin
			xc = (x0+x1)*0.5
			xc_n.append(xc)
			print(xi,'---Nx:---',Nx)
			for yi in range(Ny):
				# print(yi,'---Ny:---',Ny)
				y0 = self.ylo+dY*yi #down coord of bin
				y1 = self.ylo+dY*(yi+1) #down coord of bin
				yc = (y0+y1)*0.5
				yc_n.append(yc)
				for zi in range(Nz):
					# print(zi,'---Nz:---',Nz)
					z0 = self.zlo+dZ*zi #down coord of bin
					z1 = self.zlo+dZ*(zi+1) #down coord of bin
					zc = (z0+z1)*0.5
					zc_n.append(zc)
		
					n=0 #tot mass or number in bin

					for i in range(self.atom_n):
						
						if self.mol[i]>=mol_n[0] and self.mol[i]<=mol_n[1]:
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
