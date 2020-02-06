import numpy as np
from scipy.fftpack import fft,fft2,fftshift,ifft2,ifftshift
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import tkinter



class WaveSimulation(object):

    @staticmethod
    def make_spectrum(signal, t=None):
        n = len(signal)
        if len(t) != n:
            t = np.array([i for i in range(len(n))])
        dt = np.mean(np.diff(t))
        # fs = np.linspace(-n//2,n//2,n) / dt
        ft = (t - t[n // 2]) / dt / np.max(t)
        F = np.fft.ifftshift(np.fft.fft(signal))
        Freq_content = np.abs(F)
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(t, signal)
        ax1.set_xlabel('time')
        ax1.set_ylabel('f_{signal}')
        ax2 = fig.add_subplot(212)
        ax2.plot(ft, Freq_content)
        ax2.set_xlabel('frequency')
        ax2.set_ylabel(r'$|\mathcal{F}_{signal}|$')
        plt.show()

    @staticmethod
    def rect(x):
        return np.where(np.abs(x) <= 0.5, 1, 0)

    @staticmethod
    def python_angular_spectrum_prop(E_in,x,y,z_prop,wavelen):
        #E_in - this is your 2D complex field that you wish to propagate
        #z_prop = ...   this is the distance by which you wish to propagate
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        wavenum = 2 * np.pi / wavelen
        wavenum_sq=wavenum**2
        kx = np.fft.fftfreq(E_in.shape[0], dx / (2 * np.pi))
        ky = np.fft.fftfreq(E_in.shape[1], dy / (2 * np.pi))
        # this is just for broadcasting, the "indexing" argument prevents NumPy
        # from doing a transpose (default meshgrid behaviour)
        kx, ky = np.meshgrid(kx, ky, indexing='ij', sparse=True)
        kz_sq = kx**2 + ky**2
        # we zero out anything for which this is not true, see many textbooks on
        # optics for an explanation
        mask = wavenum_sq > kz_sq
        g = np.zeros((len(kx), len(ky.T)), dtype=np.complex_)
        g[mask]=np.exp(1j * np.sqrt(wavenum_sq - kz_sq[mask]) * z_prop)
        res = np.fft.ifft2(g * np.fft.fft2(E_in))  # this is the result
        return res


    @staticmethod
    def volume_angular_spectrum_prop(E_in, x, y, z_prop, wavelen, normalized):
        """
         DESCRIPTION:
         volume_angular_spectrum_prop is an extension
         to angular_spectrum_prop from Fourier Optic by Goddman
         %% USAGE:% speckles3d = volume_angular_spectrum_prop(E_in, x, y, 3e2, 632e-6, true)
         % speckles3d = volume_angular_spectrum_prop(E_in, x, y, [0:0.01: 1].*3e2, 632e-6, true)
         %% INPUTS:%
          E_in - Feildtopropagate, notice that it should be the same  size of x, y
          x, y - 1Darrays in mm units
        z_prop - scaler / 1Darray in mm units of the z, axial location to propagate the feild
         wl - scaler, the wave length in mm
          normalized - bool.whether to normalize achaxial section of the 2D feild
          \%% OUTPUTS:% Intensity3d - 3D array of Intensity of the EM field.
        """
        s_z = len(z_prop) #;
        kx, ky, s_x, s_y = WaveSimulation.transverse_fourier_coordinates(x, y) #;
        alpha = ((kx / (2 * np.pi))* wavelen).astype(np.complex) #; % unitless normalized fourier coordinates
        beta = ((ky / (2 * np.pi))* wavelen).astype(np.complex) #; % unitless normalized fourier coordinates
        [Alpha, Beta] = np.meshgrid(alpha, beta,) # ;
        mu = ((2 * np.pi) / wavelen)* np.sqrt((Alpha** 2 + Beta**2) - 1) #;
        convolved_fourier = fftshift(fft2(E_in))#;
        Intensity3d = np.zeros((s_x, s_y, s_z))#;
        # angular    spectrum    of    propogation    distance    z
        for z_counter in range(s_z):
            z_pos = z_prop[z_counter] #;
            angs_prop_z = np.exp(1* mu* z_pos)#;
            E_out = ifft2(ifftshift(convolved_fourier* angs_prop_z))#;
            Iout = np.abs(E_out)** 2#;
            if normalized:
                Iout = Iout/ max(Iout[:])#; % Normalized
            Intensity3d[:,:, z_counter]=Iout

        return Intensity3d

    @staticmethod
    def transverse_fourier_coordinates(x, y):
        s_x = len(x) #;
        s_y = len(y) #;
        temp1 = np.arange(-s_x / 2,s_x / 2 ) / s_x  #;  [-0.5 0.5]     unitless
        temp2 = np.arange(-s_y / 2,s_y / 2 ) / s_y  #; % [-0.5 0.5]    unitless
        kx = temp1* (s_x * 2 * np.pi / (np.max(x) - np.min(x)))  #; % fourier    coordinates in 2    pi / mm
        ky = temp2* (s_y * 2 * np.pi / (np.max(y) - np.min(y)))  #;
        return kx, ky, s_x, s_y


    # Initialize work in mm
    def __init__(self,n=2**8,nz=2**8,*args,**kwargs):
        self.n=n
        self.nz=nz
        if len(args)>2:
            self.wav_length = args[0]  # ; % wave    length in mm
            self.NA = args[1]
            self.scatter_angle =args[2] # % % determined      by      the      material %
        else:
            self.wav_length = 2e-3  # ; % wave    length in mm
            self.NA = 0.2  # ;
            self.scatter_angle = 2 / 180 * np.pi  # % % determined      by      the      material

        self.min_focal_units = kwargs.pop('zmin', 0)
        self.max_focal_units = kwargs.pop('zmax', 1)
        self.length_x = kwargs.pop('length_x', 1)
        self.length_y = kwargs.pop('length_y', 1)
        self.x, self.y = np.arange(n)/self.n, np.arange(n)/self.n
        self.x=(self.x-np.mean(self.x))*self.length_x
        self.y = (self.y - np.mean(self.y))*self.length_y
        self.XX, self.YY = np.meshgrid(self.x, self.y)

    def creat_gaussian_beam(self, beam_width=1):
        self.RHO = np.sqrt((self.XX - np.mean(self.XX)) ** 2 + (self.YY - np.mean(self.YY)) ** 2)
        self.E0 = np.exp(-self.RHO ** 2 / beam_width)

    def diffuser(self):
        dx = self.length_x / self.n
        max_angle = ((self.wav_length / dx)) / 2

        T_diff = np.exp(1j * 2 * np.pi * np.random.rand(self.n))
        T_K = fftshift(fft2(T_diff))
        T_K = T_K * ((self.RHO / np.max(self.XX[:]) * max_angle) < self.scatter_angle)  # ; % determine    scattering    angle
        T_diff = ifft2(fftshift(T_K))
        # ; % convert    to    thin    phase    plate(meaning    takeing    only    the    phase    under    fft and not the    intensity)
        self.T_diff = np.exp(1j* np.angle(T_diff))


    def lens_phase(self,focal_length=5,**kwargs):
        nargout = kwargs.pop('nargout', 1)
        z_pos=np.linspace(self.min_focal_units,self.max_focal_units,self.nz)
        z = z_pos*focal_length
        # Lens    Phase
        #phase_of_lens = focal_length*np.ones(self.n) - np.sqrt(focal_length **2 * np.ones(self.n) - (self.XX** 2) - (self.YY**2))
        phase_of_lens = focal_length*1 - np.sqrt(focal_length **2 - (self.XX** 2) - (self.YY**2)+0j)
        phase_of_lens = phase_of_lens* (2 * np.pi / self.wav_length)
        A_lens = np.exp(-1j * phase_of_lens)
        A_lens = A_lens* (self.RHO < (np.max(self.XX[:]) *(self.NA)))
        if nargout>1:
            return A_lens,z
        else:
            return A_lens


def main():
    grid=WaveSimulation(2**8,2**8,length_x=1,length_y=1)
    normalized = 0
    grid.creat_gaussian_beam(0.05)
    A_lens,z=grid.lens_phase(focal_length=10,nargout=2)
    E_until_lens=grid.E0*A_lens
    fig1=plt.figure(1)
    plt.imshow(np.abs(E_until_lens))
    speckle3D = WaveSimulation.volume_angular_spectrum_prop(E_until_lens, grid.x, grid.y, z, grid.wav_length, normalized)
    ims=[]

    for depth in range(grid.nz-1):
        im = plt.imshow(speckle3D[:,:,depth], animated=True)
        ims.append([im])
    plt.rcParams['animation.ffmpeg_path'] = 'C:\\FFmpeg\\bin\\ffmpeg.exe'
    FFwriter = animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])

    anim = animation.ArtistAnimation(fig1, ims, interval=200, blit=True,
                                    repeat_delay=100)
    anim.save('plswork.mp4', writer=FFwriter)
    plt.show()
    print(speckle3D.shape)
    plt.figure()
    plt.imshow(speckle3D[:,:, grid.nz // 2],interpolation='bicubic')
    plt.axis('off')
    plt.figure()
    plt.imshow(np.squeeze(speckle3D[:, grid.n // 2,:].T),interpolation='bicubic')
    plt.axis('off')
    plt.show()
    #return speckle3D

def animated_propagtion():
    grid = WaveSimulation(2 ** 8, 2 ** 8, length_x=10, length_y=10, zmin=-4,zmax=10)
    normalized = 0
    grid.creat_gaussian_beam(1)
    A_lens, z = grid.lens_phase(focal_length=100, nargout=2)
    E_until_lens = grid.E0 * A_lens
    fig1 = plt.figure(1)
    ims = []
    intersity3d = np.empty((grid.n, grid.n, grid.nz))
    for depth in range(grid.nz):
        feild2d = WaveSimulation.python_angular_spectrum_prop(E_until_lens, grid.x, grid.y, z[depth], 500e-6)
        intersity3d[:, :, depth] = (np.abs(feild2d) ** 2)
        im = plt.imshow(np.abs(feild2d) ** 2, animated=True)
        ims.append([im])
    plt.rcParams['animation.ffmpeg_path'] = 'C:\\FFmpeg\\bin\\ffmpeg.exe'
    FFwriter = animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])

    anim = animation.ArtistAnimation(fig1, ims, interval=200, blit=True,
                                     repeat_delay=100)
    anim.save('now.mp4', writer=FFwriter)

def make_intersity3d(grid,feild,z):
    intersity3d = np.empty((grid.n, grid.n, grid.nz))
    for depth in range(grid.nz):
        feild2d = WaveSimulation.python_angular_spectrum_prop(feild, grid.x, grid.y, z[depth], 500e-6)
        intersity3d[:, :, depth] = (np.abs(feild2d) ** 2)
    return intersity3d

def fig1():
    grid = WaveSimulation(2 ** 8, 2 ** 8, length_x=1, length_y=1,zmin=0,zmax=2)
    grid.creat_gaussian_beam(0.05)
    A_lens, z = grid.lens_phase(focal_length=20, nargout=2)
    E_until_lens = grid.E0 * A_lens
    plt.imshow(np.abs(E_until_lens))
    plt.imshow(np.angle(E_until_lens))
    plt.show()
    intersity3d=make_intersity3d(grid,E_until_lens,z)
    print(np.argmax)
    fig1_1 = plt.figure(1)
    ax1=fig1_1.add_subplot(111) # First row, first column
    ax1.imshow((intersity3d[:,:,grid.nz//2])**(1/1.5))
    ax1.set_title("XY",fontsize=30)
    plt.axis('off')
    plt.show()
    fig1_2 = plt.figure(2)
    ax2=fig1_2.add_subplot(111) # First row, second column
    ax2.imshow((intersity3d[grid.n // 2-1, :, :].T)**1.5)
    ax2.set_title("XZ",fontsize=30)
    plt.axis('off')
    plt.show()

def fig2():
    grid = WaveSimulation(2 ** 8, 2 ** 8, length_x=1, length_y=1,zmin=0,zmax=2)
    grid.creat_gaussian_beam(0.05)
    A_lens, z = grid.lens_phase(focal_length=20, nargout=2)
    E_until_lens = grid.E0 * A_lens
    plt.imshow(np.abs(E_until_lens))
    plt.imshow(np.angle(E_until_lens))
    plt.show()
    intersity3d=make_intersity3d(grid,E_until_lens,z)
    print(np.argmax)
    fig2 = plt.figure(2)
    gs=GridSpec(4,4) # 4 rows, 4 columns
    gs.tight_layout(fig2, rect=[None, None, None, None])
    ax1=fig2.add_subplot(gs[0:2,0:2]) # First row, first column
    ax1.imshow(intersity3d[:,:,grid.nz//2])
    ax1.set_title("XY")
    plt.axis('off')
    ax2=fig2.add_subplot(gs[0:2,2:4]) # First row, second column
    ax2.imshow((intersity3d[grid.n // 2-1, :, :].T)**3)
    ax2.set_title("XZ")
    plt.axis('off')
    ax3=fig2.add_subplot(gs[2,0]) # First row, third column
    plt.axis('off')
    ax4=fig2.add_subplot(gs[2,1]) # Second row, span all columns
    plt.axis('off')
    ax5 = fig2.add_subplot(gs[2, 2])  # First row, third column
    plt.axis('off')
    ax6 = fig2.add_subplot(gs[2, 3])  # Second row, span all columns
    plt.axis('off')
    ax7 = fig2.add_subplot(gs[3, 0])  # First row, third column
    plt.axis('off')
    ax8 = fig2.add_subplot(gs[3, 1])  # Second row, span all columns
    plt.axis('off')
    ax9 = fig2.add_subplot(gs[3, 2])  # First row, third column
    plt.axis('off')
    ax10 = fig2.add_subplot(gs[3, 3])  # Second row, span all columns
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    """""
    speckle3D = main()
    window = tkinter.Tk()
    # to rename the title of the window
    window.title("z axus")
    # pack is used to show the object in the window
    label = tkinter.Label(window, text="Welcome to DataCamp's Tutorial on Tkinter!").pack()
    window.mainloop()
    w=tkinter.Label(window, from_=0, to=42)
    """""
    #main()
    fig1()
    #animated_propagtion()




def make3DGaussianKernel(shape,sigma):
    if len(shape)==0:
        raise ValueError('shape is empty,should be at list given one shape')
    else:
        nx=shape[0]
        if len(shape)==2:
            ny=shape[1]
            nz = shape[2]
        if len(shape)==1:
            ny = nx
            nz = shape[1]
        else:
            ny = nx
            nz = ny

    if len(sigma)==0:
        sigma=[1,1,1]
    else:
        if len(sigma)==1:
            sigma.extend([sigma[0], sigma[0]])
        if len(shape)==2:
            sigma.insert(1,sigma[0])
    x = np.arange(0,nx)
    x = x - np.mean(x)
    y = np.arange(0, ny)
    y = y - np.mean(y)
    z = np.arange(0, nz)
    z = z - np.mean(z)
    n1, n2, n3 = np.meshgrid(x, y, z)
    gaussKER = np.exp(-n1**2/ sigma[0]  -n2**2/ sigma[1] -n3**2/ sigma[2])
    gaussKER = gaussKER/ np.sum(gaussKER)
    return gaussKER
