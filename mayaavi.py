import numpy as np
from mayavi import mlab
from tvtk.util import ctf
from matplotlib.pyplot import cm

def mayaavi_render_sf(X,Y,Z,scalar_feild,*args, ** kwargs):
    """
    X,Y,Z - catrersian cooridaned from np.grid
    scalar_feild aka sf is the reason you call this function, same shape as X,Y,Z of course

    exmaple:
    mayaavi_render_sf(X,Y,Z,3d_np_array2represtent,True,'gray',title='Title of my life')
    exmaple result in ploting VTK 3d object where the colormap is gray from
    pyplot colormap and the title is 'Title of my life'
    """
    if len(args) > 0:
        if len(args) > 1:
            mpl_cm = bool(args[0])
            colormap=args[1]
        else:
            mpl_cm=False
    else:
        mpl_cm=True
        colormap = 'viridis'

    # dealing with key word arguments
    figtitle = kwargs.pop('title', '')
    MAGNITUDE = kwargs.pop('vmax', scalar_feild.max())
    bgcolor = kwargs.pop('bgcolor', (0, 0, 0))
    fgcolor = kwargs.pop('fbgcolor', (1, 1, 1))
    height = kwargs.pop('height', 0.95)
    line_width = kwargs.pop('height', 1.1)
    size = kwargs.pop('height', 0.5)

    # now runing mlab figure
    mlab_window = mlab.figure(size=(768, 1024), bgcolor=bgcolor, fgcolor=fgcolor)
    sf = mlab.pipeline.scalar_field(X, Y, Z, scalar_feild)
    vl = mlab.pipeline.volume(sf)
    # save the existing colormap
    c = ctf.save_ctfs(vl._volume_property)
    if mpl_cm:
        values = np.linspace(0., MAGNITUDE, len(c['rgb']))
        new_cm = cm.get_cmap(colormap, lut=len(values))([i for i in range(len(values))])
        new_cm = new_cm[:, :-1]  # discard alpha channel 'cause it differently configure via the 'alpha' key in c dictionary
        c['alpha'][-1][-1] = 1.
        for i, v in enumerate(c['rgb'][1:]):  # iteration from 2nd element to keep it  background
            v[1:] = new_cm[i + 1]
    else:
        new_cm=[255/255,230/255,10/255]
        for i, v in enumerate(c['rgb'][0:]):  # iteration from 2nd element to keep it  background
            v[1:] = new_cm
    # load the new color transfer function
    ctf.load_ctfs(c, vl._volume_property)
    # signal for update
    vl.update_ctf = True
    c = ctf.save_ctfs(vl._volume_property)
    print(c, sep='\n')
    mlab.outline()
    mlab.axes()
    mlab.title(figtitle, height=height, line_width=line_width, size=size)
    mlab.xlabel('')
    mlab.ylabel('')
    mlab.show()
