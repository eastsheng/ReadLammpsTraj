# calculating the 2D density from lammps traj
# id mol type mass x y z vx vy vz fx fy fz q
# id mol element type x y z
import numpy as np
import matplotlib.pyplot as plt
import ReadLammpsTraj as RLT
import pandas as pd
from pathlib import Path
from scipy import ndimage
from scipy.signal import convolve2d
import seaborn as sns
import fastdataing as fd

def cal_2d_density(path,f,time,atomtype_dict,dbin,timestep=1):
	# Path(path+"density/2d/").mkdir(parents=True,exist_ok=True)
	md = RLT.ReadLammpsTraj(path+f)
	interstep,atom_n,Lx,Ly,Lz = md.read_info()
	# # number of frames
	# mframe,nframe = 100,100
	# # chunk number in x,y,z
	# Nx = 1
	# Ny = int(Ly*10)
	# Nz = int(Lz*10)
	Nx,Ny,Nz = 1,int(Ly/dbin),int(Lz/dbin)
	print(Nx,Ny,Nz)
	print("Nx={},Ny={},Nz={}".format(Nx,Ny,Nz))
	# if xy, Nyy=Ny, if xz, Nyy=Nz, if yz, Nxx=Ny,Nyy=Nz
	Nxx = Ny
	Nyy = Nz
	atomtype_list = [1,2,3]
	mass_list = [15.9994,1.00797,16.043]
	atomtype_Label = list(atomtype_dict.keys())
	iframe = int(time*1e6/timestep/interstep)
	print("time =",time,"ns","\nframe =",iframe)
	mframe,nframe = iframe,iframe
	for i in range(len(atomtype_Label)):
		atom_temp = atomtype_dict[atomtype_Label[i]]
		label = atomtype_Label[i]
		rho_nframe = np.zeros((Nx,Ny,Nz))
		# print(rho_nframe)
		for j in range(mframe,nframe+1):
			mxyz = md.read_mxyz_add_mass(j,atomtype_list,mass_list)
			x,y,z,rho= md.TwoD_Density(j,mxyz,atomtype_n=atom_temp,Nx=Nx,Ny=Ny,Nz=Nz,
										mass_or_number="number",id_type="atom")
			rho_nframe = rho_nframe+rho
			print(j,"---",nframe)
		nf = nframe-mframe+1
		rho=rho_nframe/nf

		rho = rho.reshape((Nxx,Nyy))
		df_rho = pd.DataFrame(rho)
		df_rho.index = y.flatten()
		df_rho.columns = z.flatten()
	return df_rho


def average_density(path,f,time,atomtype_dict,dbin,timestep=1,interstep=1):
	atomtype_Label = list(atomtype_dict.keys())
	label = atomtype_Label[0]
	iframe = int(round(time[0],1)*1e6/timestep/interstep)
	mframe,nframe = iframe,iframe
	df_rho_list = []
	for i in range(len(time)):
		df_rho = cal_2d_density(path,f,time[i],atomtype_dict,dbin,timestep=1)
		df_rho_list.append(df_rho)
	
	average_df_rho = pd.concat(df_rho_list).groupby(level=0).mean()

	print(average_df_rho)

	average_df_rho.to_csv(path+"2D_yz_ndensity_average_"+label+"_"+str(mframe)+"_"+str(round(time[0],1))+"ns.csv")

	return


def plt_2d_density(path,time,label,nzoom=2,max_density=1.0,timestep=1,interstep=1000):
	# ---------------------variables----------------------
	iframe = int(time*1e6/timestep/interstep)
	print("time =",time,"ns","\nframe =",iframe)
	mframe,nframe = iframe,iframe

	# ---------------------read----------------------
	df_rho = pd.read_csv(path+"2D_yz_ndensity_average_"+label+"_"+str(mframe)+"_"+str(time)+"ns.csv",
		index_col=0)
	print(df_rho)
	xi = df_rho.index.astype(float)
	yi = df_rho.columns.astype(float)
	rho = df_rho.values.astype(float)

	# ---------------------plot----------------------
	m,n=rho.shape
	fig = fd.add_fig(figsize=(m*0.1,n*0.1),inout="out",size=24)
	ax = fig.add_subplot(111)
	
	# # 1. 高斯过滤:周期性边界
	# # ---------------------------------------------------
	# rho = ndimage.gaussian_filter(rho,sigma=1.5)
	# 生成周期性高斯核
	sigma = 2
	kernel_size = int(6 * sigma)
	x_kernel = np.linspace(-3 * sigma, 3 * sigma, kernel_size)
	y_kernel = np.linspace(-3 * sigma, 3 * sigma, kernel_size)
	X_kernel, Y_kernel = np.meshgrid(x_kernel, y_kernel)
	gaussian_kernel = np.exp(-(X_kernel**2 + Y_kernel**2) / (2 * sigma**2))
	gaussian_kernel /= np.sum(gaussian_kernel)
	# 使用周期性高斯核进行卷积
	rho = convolve2d(rho, gaussian_kernel, mode='same', boundary='wrap')

	# 2. 放大矩阵
	# -----------------------------------------------------
	# nzoom = 5
	m, n = rho.shape
	xn = np.linspace(float(min(xi)), float(max(xi)), int(m*nzoom))
	yn = np.linspace(float(min(yi)), float(max(yi)), int(n*nzoom))
	rho = ndimage.zoom(rho, nzoom, order=3)
	
	# 4. 归一化密度
	# -----------------------------------------------------
	All_max = np.max(rho)#normalied
	print("Maximum density value is :",All_max)
	# All_max = max_density
	rho = rho/All_max

	# 3. 添加随机噪声
	# -----------------------------------------------------
	mask = (rho > 0)
	rho[mask] += np.random.normal(-0.05, 0.05, rho.shape)[mask]

	# 4. 归一噪声后的密度
	# -----------------------------------------------------
	# All_max = np.max(rho)#normalied
	# print("Noise Maximum density value is :",All_max)
	# rho = rho/All_max

	# ---------------------pcolormesh----------------------
	xx,yy = np.meshgrid(xn,yn)
	plt.pcolormesh(xx.astype(float)*1e-1,yy.astype(float)*1e-1,rho.T,
		cmap="jet",vmin=0,vmax=1,
		shading='gouraud'
		)
	# -----------------------------------------------------
	# ax.set_xlim(0,)
	# ax.set_ylim(0,)
	# ax.set_xlabel("Y (nm)",fontweight="normal",fontsize=26)	
	# ax.set_ylabel("Z (nm)",fontweight="normal",fontsize=26)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.set_xticks([])
	ax.set_yticks([])
	# ---------------------color bar----------------------
	# position = fig.add_axes([0.215, 0.95, 0.6, 0.035])
	# cb=plt.colorbar(shrink=0.5,cax=position,orientation='horizontal')
	# cb.set_ticks([0,0.2,0.4,0.6,0.8,1.0])
	# ---------------------save fig----------------------
	plt.savefig(path+"2d_yz_density_"+str(mframe)+"_"+str(time)+"ns.png",dpi=300.0)
	plt.show()

	return

if __name__ == '__main__':
	paths = ["./"]
	f = "test.lammpstrj"
	atomtype_dict ={
	            # "water":[1,2],
	            "methane":[3,3]
	            }
	# Nx,Ny,Nz = 1,160,80
	dbin = 0.5 # nm

	time = [0.0] # 1.0 ns
	path = paths[0]
	average_density(path,f,time,atomtype_dict,dbin,timestep=1)
	label = list(atomtype_dict.keys())[0] # "methane" # methane
	plt_2d_density(path,round(time[0],1),label,nzoom=5,max_density=0.05,timestep=1)

