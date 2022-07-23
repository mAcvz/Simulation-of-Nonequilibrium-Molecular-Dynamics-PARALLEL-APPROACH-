from numpy import zeros, rint, sqrt, sum, pi, int64, float64
from numpy import random
from numpy import int32
from numpy import savetxt, column_stack, empty, str_
from pickle import dump, load, HIGHEST_PROTOCOL
from my_calcener_file import calcener
from numba import njit 


                                            
  
class LJ :

  def __init__(self, rho, mx, my, mz, gforce, nstep ):
    #-start-------------------------
    self.mb   = 1.   # reference Argon 39.95 amu
    epsb      = 1.   # reference Argon 119 K
    sigb      = 1.   # reference Argon 0.3405 nm
    self.ma   = 1.   # Xe 131.29 amu = 3.29 m_Ar   # Kr 83.80 amu = 2.1 m_Ar
    epsa      = 1.   # Xe 204 K = 1.72 eps_Ar      # Kr 165 K = 1.39 eps_Ar
    siga      = 1.   # Xe 0.3975 nm =1.17 sigma_Ar # Kr 0.3633 nm = 1.07 sigma_Ar
    self.gforce = gforce
    self.c6   = zeros(3)
    self.cf6  = zeros(3)
    self.c12  = zeros(3)
    self.cf12 = zeros(3)
    self.ecut = zeros(3)
    # potential cut-off exchange order as needed 
    # WCA cut-off in the minimum
    # self.r2cut=2.**(1./3.)*max(sig1,sig2)**2
    # standard cut-off for LJ systems
    #self.rcut = max(siga,sigb)
    #self.rcut*= 3.
    # rcut needs to be an integer number of unit cell a
    a = (4/rho)**(1./3.)
    self.dslice = 2  
    cell_lenght = self.dslice*a
    self.rcut = 3.*sigb
    if (cell_lenght < self.rcut):
        self.rcut = cell_lenght
    self.lb = mz*self.dslice
    self.l1  =  mx*self.dslice
    self.l2  =  my*self.dslice ### add -2 to assume it is enough to have one lower and one higher extra box
    self.l3  =  mz*self.dslice
    self.npart   = 4*self.l1*self.l2*self.l3
    self.Lx = mx * cell_lenght
    self.Ly = my * cell_lenght
    self.Lz = mz * cell_lenght
    self.rho = self.npart/(mx*(my-2)*mz*cell_lenght**3)    
    ndim = self.npart
    # N initial guess of average number of particles for dimensioning
    self.x       = zeros( ndim )
    self.y       = zeros( ndim )
    self.z       = zeros( ndim )
    self.m       = zeros( ndim )
    self.sp      = zeros( ndim, dtype=int64 )
    self.rx      = zeros( ndim )
    self.ry      = zeros( ndim )
    self.rz      = zeros( ndim )
    self.px      = zeros( ndim )
    self.py      = zeros( ndim )
    self.pz      = zeros( ndim )
    self.fx      = zeros( ndim )
    self.fy      = zeros( ndim )
    self.fz      = zeros( ndim )
    self.etxx    = zeros( ndim )
    self.stxx    = zeros( ndim )
    self.styy    = zeros( ndim )
    self.stzz    = zeros( ndim )
    self.stxy    = zeros( ndim )
    self.stxz    = zeros( ndim )
    self.styz    = zeros( ndim )
    # dimensioning array for nemd
    self.heatt = zeros( (      2,nstep+1), dtype=float64 )
    self.hstat = zeros( (self.lb,nstep+1), dtype=float64 )
    self.hstbt = zeros( (self.lb,nstep+1), dtype=float64 )
    self.enkat = zeros( (self.lb,nstep+1), dtype=float64 )
    self.enkbt = zeros( (self.lb,nstep+1), dtype=float64 )
    self.jmazt = zeros( (self.lb,nstep+1), dtype=float64 )
    self.gzt   = zeros( (self.lb,nstep+1), dtype=float64 )
    self.jezt  = zeros( (self.lb,nstep+1), dtype=float64 )
    self.enet  = zeros( (self.lb,nstep+1), dtype=float64 )
    #
    self.kg      = 512
    self.gcount  = zeros( (self.kg,3) )
    self.ekin    = 0.0
    self.ene     = 0.0
    self.etot    = 0.0
    #
    self.tt      = 0.0
    self.ekt     = 0.0
    self.ept     = 0.0
    self.pres    = 0.0
    #
    rmax = min( (self.Lx, self.Ly, self.Lz) )/2.
    rmax = self.rcut # to compute gdr in calcener
    self.ldel  = rmax/self.kg
    self.r2max = rmax * rmax
    self.r2cut = self.rcut**2
    # particles N = Na + Nb
    N  = self.npart
    Na = self.npart//2
    # particle species: a
    self.sp[0:Na]= 0
    self.m[0:Na] = self.ma
    # particle species: b
    self.sp[Na:N]= 1
    self.m[Na:N] = self.mb
    # a-a interaction
    sig6        = siga**6
    self.c6[0]  = 4.0*epsa*sig6;
    self.c12[0] = self.c6[0]*sig6;
    self.cf12[0]= 12.*self.c12[0];
    self.cf6[0] = 6.*self.c6[0];
    self.ecut[0]= - (self.c12[0]/self.r2cut**3-self.c6[0])/self.r2cut**3;
    # a-b interaction
    sig6        = (0.5*(siga+sigb))**6
    self.c6[1]  = 4.0*sqrt(epsa*epsb)*sig6;
    self.c12[1] = self.c6[1]*sig6;
    self.cf12[1]= 12.*self.c12[1];
    self.cf6[1] = 6.*self.c6[1];
    self.ecut[1]= - (self.c12[1]/self.r2cut**3-self.c6[1])/self.r2cut**3;
    # b-b interaction
    sig6        = sigb**6
    self.c6[2]  = 4.0*epsb*sig6;
    self.c12[2] = self.c6[2]*sig6;
    self.cf12[2]= 12.*self.c12[2];
    self.cf6[2] = 6.*self.c6[2];
    self.ecut[2]= - (self.c12[2]/self.r2cut**3-self.c6[2])/self.r2cut**3;
    # 

  
  def fcc(self):
    ax = self.Lx/self.l1
    ay = self.Ly/self.l2 
    #ay = self.Ly/(self.l2 + 2*self.dslice) 
    az = self.Lz/self.l3
    print( "# lattice lenghts (ax,ay,az) =", (ax, ay, az) )
    print( "# (l1, l2, l3) =", (self.l1,self.l2, self.l3) )
    mm = self.l1*self.l2*self.l3
    natom = 4*mm  
    print( "# number of lattice cells =", mm )
    print( "# number of particles =" , natom )
    print( "# md-box sides [Lx, Ly, Lz ]=", (self.Lx, self.Ly, self.Lz) )
    j  = 0
    xi = 0.
    yi = 0.
    zi = 0.
    delta=0.005
    
    rrx = random.normal(0., delta, natom)
    rry = random.normal(0., delta, natom)
    rrz = random.normal(0., delta, natom)
    #with open("fcc.txt", "w") as f:
    for nx in range(self.l1) :
        for ny in range(self.l2) :
            for nz in range(self.l3) :
                self.x[j] = xi + ax*nx + rrx[j]
                self.y[j] = yi + ay*ny + rry[j]             
                self.z[j] = zi + az*nz + rrz[j]
                #f.write( "  %d   %8.3f   %8.3f   %8.3f \n" % (j, self.x[j], self.y[j], self.z[j]) )
                #print( (j, self.x[j], self.y[j], self.z[j]) )
                j +=1
                self.x[j] = xi + ax*nx + rrx[j] + 0.5*ax
                self.y[j] = yi + ay*ny + rry[j] + 0.5*ay     
                self.z[j] = zi + az*nz + rrz[j]
                #f.write( "  %d   %8.3f   %8.3f   %8.3f \n" % (j, self.x[j], self.y[j], self.z[j]) )
                #print( (j, self.x[j], self.y[j], self.z[j]) )
                j +=1
                self.x[j] = xi + ax*nx + rrx[j] + 0.5*ax
                self.y[j] = yi + ay*ny + rry[j]             
                self.z[j] = zi + az*nz + rrz[j] + 0.5*az
                #f.write( "  %d   %8.3f   %8.3f   %8.3f \n" % (j, self.x[j], self.y[j], self.z[j]) )
                #print( (j, self.x[j], self.y[j], self.z[j]) )
                j +=1
                self.x[j] = xi + ax*nx + rrx[j] 
                self.y[j] = yi + ay*ny + rry[j] + 0.5*ay            
                self.z[j] = zi + az*nz + rrz[j] + 0.5*az
                #f.write( "  %d   %8.3f   %8.3f    %8.3f \n" % (j, self.x[j], self.y[j], self.z[j]) )
                #print( (j, self.x[j], self.y[j], self.z[j]) )
                j +=1
    print( "# end of initial fcc lattice construction npart =",j)
  

  def calcexy(self, N) :
    # zeroing
    # nb: rectangular PBC
    enex= 0.
    egr = 0.
    viex= 0.
    eps = 1.   # reference Argon 119 K
    sig6= 1.   # reference Argon 0.3405 nm
    #ay  = self.Ly/(self.l2+2*self.dslice)
    ywu = 0.93 + self.l2*self.Ly/(self.l2+2*self.dslice)
    ywd = 0.
    c3  = 4.0*eps*sig6
    c9  = c3*sig6
    c3 *= self.rho*pi/6.
    c9 *= self.rho*pi/45.
    cf3 = 3.*c3
    cf9 = 9.*c9
    ecut= - (c9/self.rcut**6-c3)/self.rcut**3
    # loop over particles inside selected cell
    for i in range(N):
        #dx = (self.rx[i]+0.5)*self.Lx
        dy =  self.ry[i]*self.Ly
        self.fy[i] += self.gforce
        enex += self.gforce*dy
        #dz = (self.rx[i]+0.5)*self.Lz
        dyu = ywu - dy 
        dyd = dy - ywd
        if(dyu < self.rcut) :
            rr1 = 1./dyu
            rr3 = rr1*rr1*rr1
            rr6 = rr3*rr3
            ej  = (c9*rr6 -c3)*rr3 + ecut
            vir = (cf9*rr6 -cf3)*rr3
            enex+= ej
            viex+= vir
            # forces
            self.fy[i] -= vir*rr1
        elif(dyd < self.rcut) :
            rr1 = 1./dyd
            rr3 = rr1*rr1*rr1
            rr6 = rr3*rr3
            ej  = (c9*rr6 -c3)*rr3 + ecut
            vir = (cf9*rr6 -cf3)*rr3
            enex+= ej
            viex+= vir
            # forces
            self.fy[i] += vir*rr1
            ## observable --> stress tensor
            #s1yy[i]+=vir*dyu
            #s1xy[j]+=vir*dx
            #s1yz[i]+=vir*dz
    return ( enex, viex)

  def eqmdi(self, N, Na, mx, my, mz, kt, pas, mode):
    # initial evaluation of forces, energies and virials
    self.rx  =  self.x/self.Lx
    self.rx -= rint(self.rx)
    self.ry  =  self.y/self.Ly
    self.ry -= rint(self.ry)
    self.rz  =  self.z/self.Lz
    self.rz -= rint(self.rz )
    (enep, virial)=calcener(mx, my, mz, N,self.rx,self.ry,self.rz,self.Lx,self.Ly,self.Lz,self.sp,self.r2cut,self.c12,self.c6,self.ecut,self.cf12,self.cf6,self.fx,self.fy,self.fz,self.etxx,self.stxx,self.styy,self.stzz,self.stxy,self.stxz,self.styz)
    print("test 1",enep,virial)
    
    #self.ry e self.fy sono già vettori e possiamo già inserirli nel kernel
    #(enex, virex)  = self.calcexy(N)
    #print("test 2",enex,virex)
    
    if mode == 2 : 
        # andersen thermostats: velocity rescaling
        pstd=sqrt(self.ma*kt)
        self.px[0:Na] = random.normal(0., pstd, Na)
        self.py[0:Na] = random.normal(0., pstd, Na)
        self.pz[0:Na] = random.normal(0., pstd, Na)
        pstd=sqrt(self.mb*kt)
        self.px[Na:N] = random.normal(0., pstd, N-Na)
        self.py[Na:N] = random.normal(0., pstd, N-Na)
        self.pz[Na:N] = random.normal(0., pstd, N-Na)
        vcmx  = sum(self.px)
        vcmy  = sum(self.py)
        vcmz  = sum(self.pz)
        self.px[0:Na]   -= self.ma*vcmx/(Na*self.ma+(N-Na)*self.mb)
        self.py[0:Na]   -= self.ma*vcmy/(Na*self.ma+(N-Na)*self.mb)
        self.pz[0:Na]   -= self.ma*vcmz/(Na*self.ma+(N-Na)*self.mb)
        self.px[Na:N]   -= self.mb*vcmx/(Na*self.ma+(N-Na)*self.mb)
        self.py[Na:N]   -= self.mb*vcmy/(Na*self.ma+(N-Na)*self.mb)
        self.pz[Na:N]   -= self.mb*vcmz/(Na*self.ma+(N-Na)*self.mb)
        print("# velocities sampled from maxwell distribution at timestep" , pas)
    vcmx = sum(self.px)
    vcmy = sum(self.py)
    vcmz = sum(self.pz)
    enek = 0.5*sum( (self.px**2+self.py**2+self.pz**2)/self.m )
    self.ept  = 0.
    self.ekt  = 0.
    self.pres = 0.
    return (enep/N, enek/N, virial/N, vcmx, vcmy, vcmz)   


  def eqmdr(self, N, Na, mx, my, mz, kt, pas, dt, freq):
    dth=0.5*dt
    for ip in range(freq):
        pas+= 1
        t   = pas*dt
        # momenta first 
        self.px[0:N] += self.fx[0:N]*dth
        self.py[0:N] += self.fy[0:N]*dth
        self.pz[0:N] += self.fz[0:N]*dth        
        # positions second
        self.x[0:N]  += dt*self.px[0:N]/self.m[0:N]
        self.y[0:N]  += dt*self.py[0:N]/self.m[0:N]
        self.z[0:N]  += dt*self.pz[0:N]/self.m[0:N]
        # compute forces
        self.rx[0:N] =  self.x[0:N]/self.Lx
        self.rx[0:N]-= rint(self.rx[0:N])
        self.ry[0:N] =  self.y[0:N]/self.Ly
        self.ry[0:N]-= rint(self.ry[0:N])
        self.rz[0:N] =  self.z[0:N]/self.Lz
        self.rz[0:N]-= rint(self.rz[0:N])
        (enep, virial)=calcener(mx,my,mz,N,self.rx,self.ry,self.rz,self.Lx,self.Ly,self.Lz,self.sp,self.r2cut,self.c12,self.c6,self.ecut,self.cf12,self.cf6,self.fx,self.fy,self.fz,self.etxx,self.stxx,self.styy,self.stzz,self.stxy,self.stxz,self.styz)

        #self.ry e self.fy sono già vettori e possiamo già inserirli nel kernel
        #(enex, virex)  = self.calcexy(N)
        #print("test 3",enex,virex)

        # momenta third
        self.px[0:N] += self.fx[0:N]*dth
        self.py[0:N] += self.fy[0:N]*dth
        self.pz[0:N] += self.fz[0:N]*dth   
        # one step advanced 
        vcmx = sum(self.px)
        vcmy = sum(self.py)
        vcmz = sum(self.pz)
        enek = 0.5*sum( (self.px**2+self.py**2+self.pz**2)/self.m )
        self.ekt += enek
        self.ept += enep
        self.pres+= virial       
    return (t, enep/N, enek/N, virial/N, vcmx, vcmy, vcmz )     


  def nemdi(self, N, Na, mx, my, mz, kt, dkt, pas, lther):
    tlef = 3.*(kt + dkt)
    trig = 3.*kt
    # initial evaluation of forces, energies and virials
    self.rx  =  self.x/self.Lx
    self.rx -= rint(self.rx)
    self.ry  =  self.y/self.Ly
    self.ry -= rint(self.ry)
    self.rz  =  self.z/self.Lz
    self.rz -= rint(self.rz )
    (enep, virial)=calcener(mx,my,mz,N,self.rx,self.ry,self.rz,self.Lx,self.Ly,self.Lz,self.sp,self.r2cut,self.c12,self.c6,self.ecut,self.cf12,self.cf6,self.fx,self.fy,self.fz,self.etxx,self.stxx,self.styy,self.stzz,self.stxy,self.stxz,self.styz)
    #self.ry e self.fy sono già vettori e possiamo già inserirli nel kernel
    #(enex, virex)  = self.calcexy(N)

    # initializing soret thermostats: velocity rescaling
    hista,histb,eneka,enekb,heat,gz,jmaz,jez,ene = self.therm(tlef,trig,N,Na,lther)
   #
    self.heatt[:,pas] =  (heat[:])
   #
    self.jmazt[:,pas] =  ( jmaz[:])
    self.gzt[:,pas]   =  (   gz[:])
    self.jezt[:,pas]  =  (  jez[:])
   #
    self.hstat[:,pas] =  (hista[:])
    self.hstbt[:,pas] =  (histb[:])
    self.enkat[:,pas] =  (eneka[:])
    self.enkbt[:,pas] =  (enekb[:])
    self.enet[:,pas]  =  (  ene[:])
#
    vcmx = sum(self.px)
    vcmy = sum(self.py)
    vcmz = sum(self.pz)
    enek = 0.5*sum( (self.px**2+self.py**2+self.pz**2)/self.m )
    # initializing counters and constants
    self.ept  = 0.
    self.ekt  = 0.
    self.pres = 0.
    ## print("# starting dnemd trajectory")
    ## print( "( 'pas', 'enep', 'enek', 'enet', 'heatin', 'heatout', 'vcm' )")
    ## print( (0., enep/N, enek/N, (enep+enek)/N, heat[0], heat[1], vcmx, vcmy, vcmz) )
    return (enep/N, enek/N, (enep+enek)/N, heat[0], heat[1], vcmx, vcmy, vcmz)
    
  def nemdr(self, N, Na, mx, my, mz, kt, dkt, pas, dt, freq, lther):
    tlef = 3.*(kt + dkt)
    trig = 3.*kt
    dth=0.5*dt
    for ip in range(freq):
        pas+= 1
        t   = pas*dt
        # advance one step
        # momenta first 
        self.px[0:N] += self.fx[0:N]*dth
        self.py[0:N] += self.fy[0:N]*dth
        self.pz[0:N] += self.fz[0:N]*dth        
        # positions second
        self.x[0:N]  += dt*self.px[0:N]/self.m[0:N]
        self.y[0:N]  += dt*self.py[0:N]/self.m[0:N]
        self.z[0:N]  += dt*self.pz[0:N]/self.m[0:N]
        # compute forces
        self.rx[0:N] = self.x[0:N]/self.Lx
        self.rx[0:N]-= rint(self.rx[0:N])
        self.ry[0:N] = self.y[0:N]/self.Ly
        self.ry[0:N]-= rint(self.ry[0:N])
        self.rz[0:N] = self.z[0:N]/self.Lz
        self.rz[0:N]-= rint(self.rz[0:N])
        (enep, virial)=calcener(mx,my,mz,N,self.rx,self.ry,self.rz,self.Lx,self.Ly,self.Lz,self.sp,self.r2cut,self.c12,self.c6,self.ecut,self.cf12,self.cf6,self.fx,self.fy,self.fz,self.etxx,self.stxx,self.styy,self.stzz,self.stxy,self.stxz,self.styz)
        #self.ry e self.fy sono già vettori e possiamo già inserirli nel kernel
        #(enex, virex)  = self.calcexy(N)

        # momenta third
        self.px[0:N] += self.fx[0:N]*dth
        self.py[0:N] += self.fy[0:N]*dth
        self.pz[0:N] += self.fz[0:N]*dth   
        # one step advanced 
        # soret thermostats: velocity rescaling
        hista,histb,eneka,enekb,heat,gz,jmaz,jez,ene = self.therm(tlef,trig,N,Na,lther)
       #
        self.heatt[:,pas] = (heat[:])
       #
        self.jmazt[:,pas] = ( jmaz[:])
        self.gzt[:,pas]   = (   gz[:])
        self.jezt[:,pas]  = (  jez[:])
       #
        self.hstat[:,pas] = (hista[:])
        self.hstbt[:,pas] = (histb[:])
        self.enkat[:,pas] = (eneka[:])
        self.enkbt[:,pas] = (enekb[:])
        self.enet[:,pas]  = (  ene[:])
       #
        #end inner --> single step printout
        vcmx = sum(self.px)
        vcmy = sum(self.py)
        vcmz = sum(self.pz)
        enek = 0.5*sum( (self.px**2+self.py**2+self.pz**2)/self.m )
        self.ekt += enek
        self.ept += enep
        self.pres+= virial        
        ## if (pas)%freq==0 : 
            ## fout.write (" %8.3f %9.4f %9.4f %10.7f %8.3f %8.3f  %7.2g %7.2g %7.2g \n" % \
    ## print( (t, enep/N, enek/N, (enep+enek)/N, heat[0], heat[1], vcmx, vcmy, vcmz) )
    return (t, enep/N, enek/N, (enep+enek)/N, heat[0], heat[1], vcmx, vcmy, vcmz)
   # end of ne-md run
    

  def therm(self,tlef,trig,N,Na,lther) :
    # thermostats: velocity rescaling
    nindl= zeros(N, dtype=int32 )
    nla  = zeros(self.lb)
    nlb  = zeros(self.lb)
    eca  = zeros(self.lb)
    ecb  = zeros(self.lb)
    vx   = zeros(self.lb)
    vy   = zeros(self.lb)
    vz   = zeros(self.lb)
    jmaz = zeros(self.lb)
    jmz  = zeros(self.lb)
    jez  = zeros(self.lb)
    e    = zeros(self.lb)
    ppx  = zeros(N)
    ppy  = zeros(N)
    ppz  = zeros(N)
    mvx  = zeros(N)
    mvy  = zeros(N)
    mvz  = zeros(N)
    heat = zeros(2)
    for i in range(Na) :
        li = int((self.rz[i]+0.5)*self.lb) 
        nla[li] += 1.
        nindl[i] = li
        vx[li]+= self.px[i]
        vy[li]+= self.py[i]
        vz[li]+= self.pz[i]
    for i in range(Na,N) :
        li = int((self.rz[i]+0.5)*self.lb) 
        nlb[li] += 1.
        nindl[i] = li
        vx[li]+= self.px[i]
        vy[li]+= self.py[i]
        vz[li]+= self.pz[i]
    ml = zeros(self.lb)
    for li in range(self.lb) :
        ml[li] = (nla[li]*self.m[0]+nlb[li]*self.m[Na])
    for i in range(Na) :
        li = nindl[i] 
        mvx[i]  = self.m[i] *vx[li]/ml[li]
        ppx[i]  = self.px[i]-mvx[i]
        mvy[i]  = self.m[i] *vy[li]/ml[li]
        ppy[i]  = self.py[i]-mvy[i]
        mvz[i]  = self.m[i] *vz[li]/ml[li]
        ppz[i]  = self.pz[i]-mvz[i]
        eca[li]+=(ppx[i]**2 + ppy[i]**2 + ppz[i]**2) 
    eca /=self.m[0]
    for i in range(Na,N) :
        li = nindl[i] 
        mvx[i]  = self.m[i] *vx[li]/ml[li]
        ppx[i]  = self.px[i]-mvx[i]
        mvy[i]  = self.m[i] *vy[li]/ml[li]
        ppy[i]  = self.py[i]-mvy[i]
        mvz[i]  = self.m[i] *vz[li]/ml[li]
        ppz[i]  = self.pz[i]-mvz[i]
        ecb[li]+=(ppx[i]**2 + ppy[i]**2 + ppz[i]**2) 
    ecb /=self.m[Na]
    # thermostatting 'lther' layers on the left side of the MD-box
    eclef= zeros(self.lb)
    ecrig= zeros(self.lb)
    nlef = zeros(self.lb)
    nrig = zeros(self.lb)
    lfact= zeros(self.lb)
    rfact= zeros(self.lb)
    eclef[0:lther]= eca[0:lther] + ecb[0:lther]
    nlef[0:lther] = nla[0:lther] + nlb[0:lther]
    lfact[0:lther]=     tlef*(nlef[0:lther]-1.)/eclef[0:lther]
    heat[0]       = sum(    (lfact[0:lther]-1.)*eclef[0:lther])
    eca[0:lther] *= lfact[0:lther]
    ecb[0:lther] *= lfact[0:lther]
    lfact[0:lther]=sqrt(lfact[0:lther])
#
    for i in range(N) :
        li = nindl[i]
        if  li < lther :
            ppx[i]*= lfact[li]
            ppy[i]*= lfact[li]
            ppz[i]*= lfact[li]
            self.px[i]  = ppx[i]+mvx[i]
            self.py[i]  = ppy[i]+mvy[i]
            self.pz[i]  = ppz[i]+mvz[i]
# put back all analysis
    for i in range(N) :
        self.etxx[i] += (ppx[i]**2 + ppy[i]**2 + ppz[i]**2)/self.m[i]
    for i in range(Na) :
        li = nindl[i] 
        e[li]    += self.etxx[i]
        jmaz[li] += ppz[i]
        jez[li] += 0.5*(ppz[i]*self.etxx[i] + self.stxz[i]*ppx[i] + self.styz[i]*ppy[i] + self.stzz[i]*ppz[i])/self.m[i]
    for i in range(Na,N) :
        li = nindl[i] 
        e[li]    += self.etxx[i]
        jez[li] += 0.5*(ppz[i]*self.etxx[i] + self.stxz[i]*ppx[i] + self.styz[i]*ppy[i] + self.stzz[i]*ppz[i])/self.m[i]
    #    
    return (nla, nlb ,eca, ecb, heat, vz, jmaz, jez, e)

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
      dy-= rint(dy)
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
