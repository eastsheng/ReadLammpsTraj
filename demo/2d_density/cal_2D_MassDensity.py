# calculating the 2D density from lammps traj
# id mol type mass x y z vx vy vz fx fy fz q
# id mol element type x y z
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import ReadLammpsTraj as RLT
import pandas as pd
import seaborn as sns
from scipy import ndimage
def plt_2d_density():
	path = "./"
	f = 'test.lammpstrj'
	# Path(path+"2D_density/").mkdir(parents=True,exist_ok=True)
	md = RLT.ReadLammpsTraj(path+f)
	md.read_info()
	# # number of frames
	mframe,nframe = 0,0
	# # chunk number in x,y,z
	Nx = 1
	Ny = 60
	Nz = 60

	# if xy, Nyy=Ny, if xz, Nyy=Nz, if yz, Nxx=Ny,Nyy=Nz
	Nxx = Ny
	Nyy = Nz 

	atomtype_dict ={
	            # "water":[1,2,3],
	            "methane":[3,3]
	            }

	atomtype_list = [1,2,3]
	mass_list = [15.9994,1.00797,16.043]
	atomtype_Label = list(atomtype_dict.keys())
	# -------------------
	plt.rc('font', family='Times New Roman', size=22)
	fig = plt.figure(figsize=(8,8))	
	for i in range(len(atomtype_Label)):
		atom_temp = atomtype_dict[atomtype_Label[i]]
		label = atomtype_Label[i]
		rho_nframe = np.zeros((Nx,Ny,Nz))
		# print(rho_nframe)
		for j in range(mframe,nframe+1):
			mxyz = md.read_mxyz_add_mass(j,atomtype_list,mass_list)
			x,y,z,rho= md.TwoD_Density(mxyz,atomtype_n=atom_temp,Nx=Nx,Ny=Ny,Nz=Nz,mass_or_number="mass")
			rho_nframe = rho_nframe+rho
			# print(x,z)
			print(j,"---",nframe)

		nf = nframe-mframe+1
		rho=rho_nframe/nf
		rho = rho.reshape((Nxx,Nyy))
		df_rho = pd.DataFrame(rho)
		df_rho.index = y.flatten()
		df_rho.columns = z.flatten()
		print(df_rho)
		df_rho.to_csv(path+"2D_yz_ndensity_average_"+label+".csv")

		xi = df_rho.index.tolist()
		zi = df_rho.columns.tolist()#[1:]
		print(len(xi),len(zi))
		print(df_rho)
		rho = df_rho.values#[:,1:]
		rho = ndimage.gaussian_filter(rho,sigma=2)
		print(rho.shape)
		# ---------------------plot----------------------
		ax = fig.add_subplot(111)
		All_max = np.max(rho)#normalied
		print(All_max)
		# All_max = 2.698075423398127
		rho = rho/All_max
		# print(rho.shape)
		# -----------------------------------------------------
		x,z = np.meshgrid(xi,zi) #need modify 
		x = x.astype(np.float64)
		z = z.astype(np.float64)
		# print(type(x),type(z))
		# plt.pcolormesh(z,x,rho.T,cmap='jet',shading='gouraud')#need modify
		# plt.contourf(z,x,rho.T,cmap='viridis',alpha=0.5,vmin=0,vmax=1,)#need modify
		plt.pcolormesh(x,z,rho,cmap='jet',vmin=0,vmax=1,shading='gouraud')#need modify
		# -----------------------------------------------------

		# ax.set_xlim(0,35)
		# ax.set_ylim(0,35)
		# ax.set_xticks([0,10,20,30])
		# ax.set_yticks([0,10,20,30])
		ax.set_ylabel("Z(Å)",fontweight="bold")
		ax.set_xlabel("Y(Å)",fontweight="bold")	
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)
		# -----------------------------------------------------
		position = fig.add_axes([0.215, 0.925, 0.6, 0.035])
		cb=plt.colorbar(shrink=0.5,cax=position,orientation='horizontal')
		cb.set_ticks([0,0.2,0.4,0.6,0.8,1.0])
		# plt.cm.ScalarMappable.set_clim((0,1))
	plt.savefig(path+"2d_density.png",dpi=300.0)
	plt.show()

	return



if __name__ == '__main__':
	plt_2d_density()
