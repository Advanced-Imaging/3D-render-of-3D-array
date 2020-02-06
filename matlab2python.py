#Requirement - moudle import
import numpy as np
from numpy.random import uniform
from scipy.ndimage.filters import convolve
import os,time
import scipy.io as sio
import plotly.graph_objects as go
from plot_fun import Visual3D
print(os.path.dirname(__file__))
path=r"D:\Nir\Master\Experiment\saved_data"
# Find files from dir
matlab_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mat')]

with open("list_of_mat_files.txt", "w") as text_file:
    for matfile in matlab_files:
        print(matfile,'\t',"last modified: %s" % time.ctime(os.path.getmtime(matfile)) ,end='\n',file=text_file)

#selected_mat='scanned_nile_read_bead_py.mat'
selected_mat='scanned_nile_read_bead_forloop_py.mat'
def open_mat_files():
    # per file
    for p in matlab_files:
        mat_fname = sio.loadmat(p)
        print(mat_fname)
    pass

matfile_name=os.path.join(path, selected_mat)
mat_f = sio.loadmat(matfile_name)
matfile=mat_f['cc2']
PSF = matfile.shape
#world = Visual3D()
#world.plot_cube(matfile)
#print(mat_f)


#We order all the directories by name
files = [t for t in os.listdir(path)]
files.sort() #the os.listdir function do not give the files in the right order so we need to sort them
