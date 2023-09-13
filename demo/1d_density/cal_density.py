# calculating the density from lammps traj
# id mol type mass x y z vx vy vz fx fy fz q
import numpy as np
import matplotlib.pyplot as plt
import ReadLammpsTraj as RLT
from pathlib import Path
import periodictable as pt

def cal_density():
	path = "./"
	f = "1000.lammpstrj"
	# lammpsdata = "1_npt_dissociation.data"
	Path(path+"imgs/").mkdir(parents=True,exist_ok=True)

	md = RLT.ReadLammpsTraj(path+f)
	md.read_info()
	# # number of frames
	mframe,nframe = 1, 1
	# # chunk number in z
	Nz = 100
	id_range = [1,10000]
	# # element mass
	# O_mass = pt.elements.symbol('O').mass
	# H_mass = pt.elements.symbol('H').mass
	# C_mass = pt.elements.symbol('C').mass
	# mass_dict = {1:O_mass,2:H_mass,3:C_mass+4*H_mass,4:C_mass+4*H_mass}
	mass_dict = {}
	# mass_dict = RLT.read_mass(path+lammpsdata)

	direction = "z"

	Calculate = True #True#False

	if Calculate==True:
		rho_nframe = np.zeros(Nz).reshape(Nz,1)
		for i in range(mframe,nframe+1):
			position = md.read_mxyz(i)
			# z, rho = md.oneframe_alldensity(position,Nz,mass_dict,density_type="mass",direction=direction)
			z, rho = md.oneframe_moldensity(position,Nz,id_range,mass_dict,id_type="atom",density_type="mass",direction=direction)

			rho_nframe = rho_nframe+rho

			print(i,"---",nframe)
		nf = (nframe-mframe+1)
		rho=rho_nframe/nf #all
		print("Average density =",rho.mean(),"g/mL")
	#---- end ----------------------------------------------
	plt.rc('font', family='Times New Roman', size=26)
	fig = plt.figure(figsize=(12,8))
	fig.subplots_adjust(bottom=0.2,left=0.2)
	ax=fig.add_subplot(111)
	ax.plot(z,rho,'r--',label='All',linewidth=2)

	# ax.legend(loc="center")
	ax.legend(loc="best")

	ax.set_ylabel('Density(g/mL)',fontweight='bold',size=32)
	ax.set_xlabel('z(Å)',fontweight='bold',size=32)	
	# ax.set_ylim(-0.05,1.5)

	# plt.savefig(path+"imgs/density.png",dpi=300)
	plt.show()


if __name__ == '__main__':
	cal_density()


	