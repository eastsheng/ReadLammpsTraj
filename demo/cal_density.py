# calculating the density from lammps traj
# id mol type mass x y z vx vy vz fx fy fz q
import numpy as np
import matplotlib.pyplot as plt
import ReadLammpsTraj as RLT
from pathlib import Path

def cal_density():
	path = "./"
	f = "1000.lammpstrj"

	Path(path+"density/").mkdir(parents=True,exist_ok=True)

	md = RLT.ReadLammpsTraj(path+f)
	md.read_info()
	# # number of frames
	mframe,nframe = 1, 1
	# # chunk number in z
	Nz = 100
	mol_n = [1,1000]
	# 分子序号

	Calculate = True #True#False

	if Calculate==True:
		rho_nframe = np.zeros(Nz).reshape(Nz,1)
		for i in range(mframe,nframe+1):
			position = md.read_mxyz(i)
			z, rho = md.oneframe_alldensity(position,Nz,density_type="mass")
			# z, rho = md.oneframe_moldensity(position,Nz,mol_n=mol_n,id_type="mol",density_type="mass")

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

	plt.savefig(path+"density/density.png",dpi=300)
	plt.show()


if __name__ == '__main__':
	cal_density()


	