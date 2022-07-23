from numpy import zeros, pi, sqrt, rint, int64, float64,int32
from numpy import savetxt, column_stack, empty, str_
from numpy import pi
from numpy import random
from pickle import dump, load, HIGHEST_PROTOCOL

def read_input(self, N, conf_in='conf_in.b'): #, mom_in='mom_in'):
    with open(conf_in, 'rb') as ftrj:
        (Nr, pas) = load(ftrj)
        if N!=Nr :
            #print(' reading %d particle configuration from step %d' % (Nr,pas) )
        #else :
            print(' ??? reading %d particle configuration expected %d' % (Nr,N) )
        ( self.x,  self.y,  self.z ) = load( ftrj)
        ( self.px, self.py, self.pz) = load( ftrj)
    return pas

def write_input(self, N, pas, conf_out='conf_in.b'): #, mom_out='mom_in'):
    with open(conf_out, 'wb') as ftrj:
        dump( (N, pas) , ftrj, HIGHEST_PROTOCOL)
        dump( ( self.x,  self.y,  self.z ), ftrj, HIGHEST_PROTOCOL)
        dump( ( self.px, self.py, self.pz), ftrj, HIGHEST_PROTOCOL)
               
def dumpxyz(self, N, Na, pas, dumpf):
    dx = self.x/self.Lx
    dx-= rint(dx)
    dy = self.y/self.Ly
    #dy-= rint(dy)
    self.y = dy*self.Ly
    dz = self.z/self.Lz
    dz-= rint(dz)
    ar = empty(N,(str_,2))
    sig=3.4 # in Angstroem for argon   
    ar[0:Na] = "Ar"
    ar[Na:N] = "Ar"
    dx *= sig*self.Lx
    dy *= sig*self.Ly
    dz *= sig*self.Lz
    dumpf.write( " %d \n" % N ) 
    dumpf.write( " %d \n" % pas ) 
    for i in range(N):
        dumpf.write( "%s  %10.5f   %10.5f  %10.5f\n" % (ar[i], dx[i], dy[i], dz[i]) )
            
def writexyz(self, N, Na):
    dx = self.x/self.Lx
    dx-= rint(dx)
    self.x = dx*self.Lx
    dy = self.y/self.Ly
    dy-= rint(dy)
    self.y = dy*self.Ly
    dz = self.z/self.Lz
    dz-= rint(dz)
    self.z = dz*self.Lz
    ar = empty(N,(str_,2))
    sig=3.4 # in Angstroem for argon   
    ar[0:Na] = "Ar"
    ar[Na:N] = "Ar"
    dx *= sig*self.Lx
    dy *= sig*self.Ly
    dz *= sig*self.Lz
    rout = column_stack( (ar, dx, dy, dz) )
    savetxt('conf.xyz', rout, fmt=(' %s', '%s ','%s','%s'), \
    header=(' %d \n' % N ), comments=' ' )          

def write_out(self, N, Na, tstep, gdr_out='gdr.out'):
    V = zeros(self.kg) 
    r = zeros(self.kg)
    g = zeros( (self.kg,3) ) 
    for lm in range(self.kg) :
        V[lm] = 4./3.*pi*(self.ldel**3)*(3*lm*lm +3*lm + 1); 
        r[lm] = (lm+0.5)*self.ldel
    g[:,0] = self.gcount[:,0]/(V*(Na-1)*tstep*self.rho*Na/N);
    g[:,1] = self.gcount[:,1]/(2.*V*Na*(N-Na)*tstep*self.rho/N);
    g[:,2] = self.gcount[:,2]/(V*(N-(Na-1))*tstep*self.rho*(N-Na)/N);
    gout = column_stack( (r, g) )
    savetxt(gdr_out, gout , fmt=('%10.5g ','%12.7g','%12.7g','%12.7g'), \
    header="    'r'     'gaa'     'gab'     'gbb' " )          
