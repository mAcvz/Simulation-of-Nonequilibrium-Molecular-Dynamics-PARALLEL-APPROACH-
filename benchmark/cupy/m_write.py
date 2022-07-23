import numpy as np 
from pickle import dump, load, HIGHEST_PROTOCOL

def read_input(self, N, conf_in='conf_in.b'): #, mom_in='mom_in'):
    with open(conf_in, 'rb') as ftrj:
        (Nr, pas, self.zwr, self.fwall) = load(ftrj)
        if N!=Nr :
            #print(' reading %d particle configuration from step %d' % (Nr,pas) )
        #else :
            print(' ??? reading %d particle configuration expected %d' % (Nr,N) )
        ( self.x,  self.y,  self.z ) = load( ftrj)
        ( self.px, self.py, self.pz) = load( ftrj)
    return pas

def write_input(self, N, pas, conf_out='conf_in.b'): #, mom_out='mom_in'):
    with open(conf_out, 'wb') as ftrj:
        dump( (N, pas, self.zwr, self.fwall) , ftrj, HIGHEST_PROTOCOL)
        dump( ( self.x,  self.y,  self.z ), ftrj, HIGHEST_PROTOCOL)
        dump( ( self.px, self.py, self.pz), ftrj, HIGHEST_PROTOCOL)
               
def dumpxyz(self, N, pas, dumpf):
    dx = self.x/self.Lx
    dx-= np.rint(dx)
    dy = self.y/self.Ly
    #dy-= np.rint(dy)
    dz = self.z/self.Lz
    #dz-= np.rint(dz)
    ar = np.empty(N,(np.str_,2))
    sig=3.4 # in Angstroem for argon   
    ar[0:N] = "Ar"
    dx *= sig*self.Lx
    dy *= sig*self.Ly
    dz *= sig*self.Lz
    dumpf.write( " %d \n" % N ) 
    dumpf.write( " %d %10.5f  %10.5f  %10.5f  %10.5f \n" % (pas, sig*self.zwl, sig*self.zwr, sig*self.ywb, sig*self.ywt) ) 
    for i in range(N):
        dumpf.write( "%s  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f  %10.5f\n" % (ar[i],dz[i],dx[i],dy[i],self.pz[i],self.px[i],self.py[i]) )
            
def writexyz(self, N, filexyz='conf.xyz'):
    dx = self.x/self.Lx
    dx-= np.rint(dx)
    dy = self.y/self.Ly
    #dy-= np.rint(dy)
    dz = self.z/self.Lz
    #dz-= np.rint(dz)
    ar = np.empty(N,(np.str_,2))
    sig=3.4 # in Angstroem for argon   
    ar[0:N] = "Ar"
    dx *= sig*self.Lx
    dy *= sig*self.Ly
    dz *= sig*self.Lz
    rout = np.column_stack( (ar, dx, dy, dz) )
    np.savetxt(filexyz, rout, fmt=(' %s', ' %s',' %s',' %s'), \
    header=(" %d \n walls: %10.5f  %10.5f  %10.5f  %10.5f " % (N, sig*self.zwl, sig*self.zwr, sig*self.ywb, sig*self.ywt)) )          

def write_out(self, N, tstep, gdr_out='gdr.out'):
    V = np.zeros(self.kg) 
    r = np.zeros(self.kg)
    g = np.zeros( (self.kg) ) 
    for lm in range(self.kg) :
        V[lm] = 4./3.*np.pi*(self.ldel**3)*(3*lm*lm +3*lm + 1); 
        r[lm] = (lm+0.5)*self.ldel
    g[:] = self.gcount[:]/(V*(N-1)*tstep*self.rho);
    gout = np.column_stack( (r, g) )
    np.savetxt(gdr_out, gout , fmt=('%10.5g ','%12.7g'), header="    'r'         'gaa'    " )          
