
from tkinter import simpledialog,messagebox,Tk
from utils_func import make3DGaussianKernel
import numpy as np
from mayaavi import mayaavi_render_sf

from scipy.ndimage import convolve as convolve3d



parent = Tk() # Create the object
parent.overrideredirect(1) # Avoid it appearing and then disappearing quickly
#parent.iconbitmap("PythonIcon.ico") # Set an icon (this is optional - must be in a .ico format)
#parent.withdraw() # Hide the window as we do not want to see this one
save = messagebox.askyesno('Random distirbute', 'Are you saveing the output in python pickle?', parent=parent) # Yes / No
if save:
    random = messagebox.askyesno('Random distirbute', 'Do you want random point objects?', parent=parent)  # Yes / No
    PSF = messagebox.askyesno('Point like', 'Are you illustate 3d PSF', parent=parent)  # Yes / No


load = messagebox.askyesno('Point like', 'Are you loading from python pickle file', parent=parent) # Yes / No


if  load:
    import pickle as pkl
    path = r"D:\Nir\Master\thesis\images n figure\\"
    filename = path + "3d_point_objs"
    with open(filename, 'rb') as f:
        data,blurred = pkl.load(f)
    vol_shape = data.shape
    MAGNITUDE=data.max()
    X, Y, Z = np.mgrid[0:vol_shape[0], 0:vol_shape[1], 0:vol_shape[2]]
    mayaavi_render_sf(X, Y, Z, data,title='Point Sources In Volume')
    mayaavi_render_sf(X, Y, Z, blurred,'const',title='Point Sources PSF''s Volume')
else:
    vol_shape = (100, 100, 200)
    X, Y, Z = np.mgrid[0:vol_shape[0], 0:vol_shape[1], 0:vol_shape[2]]


    def make_ball(volume, radius=3):
        r = np.sqrt((X - X.mean()) ** 2 + (Y - Y.mean()) ** 2 + (Z - Z.mean()) ** 2)
        data = np.zeros(shape=vol_shape, dtype='bool')
        data[r < radius] = True
        pass


    if random:
        p1 = 0.99999  # probability of choice
        data = np.random.choice([False, True], size=vol_shape, p=[p1, 1 - p1])
        data = np.random.choice([0, 1], size=(100, 100, 200), p=[p1, 1 - p1])


    else:
        # randomly select a point coordinates
        # and then iterate over axes 0,1,2 to creat replica of them
        # according to roll location
        # here:
        # rolls = (-4, 4, -20)
        # axes = (0, 1, 2)
        # sum them up binary and then cast them
        # point_location=(np.random.rand(3)*np.asarray(vol_shape)).astype('int')
        # data=np.zeros(shape=vol_shape,dtype='bool')
        # data[tuple(point_location)]=True
        # data[vol_shape[0]//2,vol_shape[1]//2,vol_shape[2]//2]=True
        r = np.sqrt((X - X.mean()) ** 2 + (Y - Y.mean()) ** 2 + (Z - Z.mean()) ** 2)
        data = np.zeros(shape=vol_shape, dtype='bool')
        radius = 2
        data[r < radius] = True
        replica = []
        axes = (0, 1, 2)
        number_of_balls = 10
        for i in range(number_of_balls):
            point_location = (np.random.rand(3) * np.asarray(vol_shape)).astype('int')
            replica.append(np.roll(data, tuple(point_location), axis=axes))
        for i in range(len(replica)):
            data += replica[i]
    data = data.astype('float64')
    MAGNITUDE = 255
    data *= MAGNITUDE
    blurred = data.copy()
    if PSF:
        kernel = make3DGaussianKernel([50, 50, 100], [30, 30, 100])
        blurred = convolve3d(blurred, kernel, mode='constant')

    if save:
        import pickle as pkl

        path = r"D:\Nir\Master\thesis\images n figure\\"
        filename = path + "3d_point_objs"
        fileObject = open(filename, 'wb')
        pkl.dump([data, blurred], fileObject)
        fileObject.close()
    mayaavi_render_sf(X,Y,Z,data)
    mayaavi_render_sf(X,Y,Z,blurred)

