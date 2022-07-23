from numpy import zeros, ones, rint, sqrt, sum, pi,amax
from numpy import random,append
from numpy import int32,float64
from numpy import logical_and

class LJ :

  def __init__(self, rho, gforce, mx, my, mz, zbuffer, nstep ):
    #-start-------------------------
    self.m  = 1.   # Xe 131.29 amu = 3.29 m_Ar   # Kr 83.80 amu = 2.1 m_Ar
    eps     = 1.   # Xe 204 K = 1.72 eps_Ar      # Kr 165 K = 1.39 eps_Ar
    sig     = 1.   # Xe 0.3975 nm =1.17 sigma_Ar # Kr 0.3633 nm = 1.07 sigma_Ar
    # potential cut-off: 3*sigma or an integer number of unit cell a 
    # WCA cut-off in the minimum
    # self.r2cut=2.**(1./3.)*max(sig1,sig2)**2
    # standard cut-off for LJ systems
    a = (4/rho)**(1./3.)
    self.dslice = 2  # analysis over slices of width a 
    cell_lenght = self.dslice*a
    self.rcut = 3.*sig
    if (cell_lenght < self.rcut):
        self.rcut = cell_lenght
    self.lb   =  mz*self.dslice
    self.l1   =  mx*self.dslice
    self.ybuff= 2  ### assume one lower and one higher additional box in input my
    self.l2   = (my-self.ybuff)*self.dslice 
    self.zbuff= zbuffer  ### additional space for moving wall included in input mz
    self.l3   = (mz-self.zbuff)*self.dslice
    self.npart= 4*self.l1*self.l2*self.l3
    self.Lx   = mx * cell_lenght
    self.Ly   = my * cell_lenght
    self.Lz   = mz * cell_lenght
    # density estimate 
    self.rho  = self.npart/(mx*(my-self.ybuff)*(mz-self.zbuff)*cell_lenght**3)
    # wall potential epsilon=sigma=1
    self.c3  = 4.0*self.rho*pi/6.
    self.c9  = self.c3*self.rho*pi/45
    self.cf3 = 3.*self.c3
    self.cf9 = 9.*self.c9
    ## purely repulsive: cut off at potential minimum 
    self.rwcut = (3.*self.c9/self.c3)**(1./6.)
    ## positiong wall
    self.zwr = a*(self.l3 + self.dslice - 0.25) + self.rwcut
    self.zwl = a*(self.dslice + 0.25) - self.rwcut
    self.ywt = a*(self.l2 + self.dslice - 0.25) + self.rwcut
    self.ywb = a*(self.dslice + 0.25) - self.rwcut
    self.fwr = 0.
    self.fwl = 0.
    self.fwt = 0.
    self.fwb = 0.
    self.fwall = 0.
    #self.rwcut *= 10    # uncomment for an attractive wall
    self.ewcut= - (self.c9/self.rwcut**6-self.c3)/self.rwcut**3
    self.gforce= gforce
    self.gamma = 0.5*self.Lx*self.Ly
    self.mode  = 0
    #
    ndim = self.npart
    ncells=mx*my*mz
    self.np   = zeros(    ncells, dtype=int32 )
    self.indc = zeros(self.npart, dtype=int32)
    self.indcp= zeros(      ndim, dtype=int32 )
    self.indp = zeros(  ncells+1, dtype=int32)
    # indexing neighbour cells of selected one
    self.vcx1 = zeros(27, dtype=int32)
    self.vcy1 = zeros(27, dtype=int32)
    self.vcz1 = zeros(27, dtype=int32)
    # indexing neighbour cells of selected one
    k = 0
    self.vcx1 = zeros(27, dtype=int32)
    self.vcy1 = zeros(27, dtype=int32)
    self.vcz1 = zeros(27, dtype=int32)
    for i in range(-1,2):
        for j in range(-1,2):
            for l in range(-1,2):
                self.vcx1[k]= i
                self.vcy1[k]= j
                self.vcz1[k]= l
                k+=1
    # N initial guess of average number of particles for dimensioning
    self.x       = zeros( ndim )
    self.y       = zeros( ndim )
    self.z       = zeros( ndim )
    self.rx      = zeros( ndim )
    self.ry      = zeros( ndim )
    self.rz      = zeros( ndim )
    self.rcx     = zeros( ndim )
    self.rcy     = zeros( ndim )
    self.rcz     = zeros( ndim )
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
    self.heatt = zeros( (      3,nstep+1), dtype=float64 )
    self.hstat = zeros( (self.lb,nstep+1), dtype=float64 )
    #self.hstbt = zeros( (self.lb,nstep+1), dtype=float64 )
    self.enkat = zeros( (self.lb,nstep+1), dtype=float64 )
    #self.enkbt = zeros( (self.lb,nstep+1), dtype=float64 )
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
    self.lload   = 0.0
    self.rload   = 0.0
    self.tload   = 0.0
    self.bload   = 0.0
    #
    rmax = min( (self.Lx, self.Ly, self.Lz) )/2.
    rmax = self.rcut # to compute gdr in calcener
    self.ldel  = rmax/self.kg
    self.r2max = rmax * rmax
    self.r2cut = self.rcut**2
    # particles N = Na + Nb
    N  = self.npart
    # a-a interaction
    sig6     = sig**6
    self.c6  = 4.0*eps*sig6;
    self.c12 = self.c6*sig6;
    self.cf12= 12.*self.c12;
    self.cf6 = 6.*self.c6;
    self.ecut= - (self.c12/self.r2cut**3-self.c6)/self.r2cut**3;
    #
    # RANDOM WITH FIXED SEED 
    #self.rng = random.default_rng(12345)
    
    
  def fcc(self):
    ax = self.Lx/self.l1
    ay = self.Ly/(self.l2 + self.ybuff*self.dslice)
    az = self.Lz/(self.l3 + self.zbuff*self.dslice) 
    print( "# lattice lenghts (ax,ay,az) =", (ax, ay, az) )
    print( "# (mx, my, mz) =", (self.l1,self.l2, self.l3) )
    mm = self.l1*self.l2*self.l3
    natom = 4*mm  
    print( "# number of lattice cells =", mm )
    print( "# number of particles =" , natom )
    print( "# md-box sides [Lx, Ly, Lz ]=", (self.Lx, self.Ly, self.Lz) )
    j  = 0
    xi = 0.25*ax
    yi = self.dslice*ay + 0.25*ay
    zi = self.dslice*az + 0.25*az
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
    #

  def cells(self, mx, my, mz, N ):
    # nb: reduced coordinates for orthorombic box, option with/without PBC
    self.rx  = self.x/self.Lx
    self.rx -= rint(self.rx)  # si periodicity along x
    self.ry  = self.y/self.Ly
    #self.ry -= rint(self.ry) # no periodicity along y
    self.rz  = self.z/self.Lz
    #self.rz -= rint(self.rz) # no periodicity along z
    #
    ncells=mx*my*mz
    self.np[:] = 0  # zeros(ncells, dtype=int32)
    #self.indc = zeros(N, dtype=int32)
    for i in range(N):
        vcx=int(mx*(self.rx[i]+0.5)) #PBC no->int(my*(self.ry[i]    ))
        vcy=int(my*(self.ry[i]    )) #PBC si->int(my*(self.ry[i]+0.5))
        vcz=int(mz*(self.rz[i]    )) #PBC si->int(mz*(self.rz[i]+0.5))
        # cell index
        c = mz*(my*vcx+vcy)+vcz
        self.indc[i]=c
        self.np[c] += 1
    self.indp[0] = 0 # zeros(ncells+1, dtype=int32)
    for c in range(0,ncells) :
        self.indp[c+1] = self.indp[c] + self.np[c]
    for i in range(N):
        c=self.indc[i]
        self.rcx[self.indp[c]] = (self.rx[i]+0.5)*self.Lx #PBC no->(self.rx[i]    )*self.Lx 
        self.rcy[self.indp[c]] = (self.ry[i]    )*self.Ly #PBC si->(self.ry[i]+0.5)*self.Ly 
        self.rcz[self.indp[c]] = (self.rz[i]    )*self.Lz #PBC si->(self.rz[i]+0.5)*self.Lz 
        self.indcp[self.indp[c]] = i
        self.indp[c] += 1
    # need to reconstruct index list
    self.indp[0]=0
    for c in range(0,ncells) :
        self.indp[c+1] = self.indp[c] + self.np[c]

  def calcener(self, mx, my, mz, N ) :
    # zeroing
    # nb: rectangular PBC
    ene=0.
    vip=0.
    e1xx = zeros(N)
    s1xx = zeros(N)
    s1yy = zeros(N)
    s1zz = zeros(N)
    s1xy = zeros(N)
    s1xz = zeros(N)
    s1yz = zeros(N)
    f1x= zeros(N)
    f1y= zeros(N)
    f1z= zeros(N)
    #
    rc1x = zeros(27*amax(self.np))
    rc1y = zeros(27*amax(self.np))
    rc1z = zeros(27*amax(self.np))
    # Loop over Cells
    for vcx in range(mx):
        for vcy in range(1,my-1):
            for vcz in range(1,mz-1):
                c = mz*(my*vcx+vcy)+vcz  
                # loop over particles inside selected cell
                '''
                    none
                '''
                # loop over particles in neighbour cells 
                #rc1x = zeros(1) 
                #rc1y = zeros(1)
                #rc1z = zeros(1)
                ik = 0
                for k in range(27) :
                    wcx=vcx + self.vcx1[k]
                    wcy=vcy + self.vcy1[k]
                    wcz=vcz + self.vcz1[k]
                    # Periodic boundary conditions 
                    shiftx = 0.
                    if (wcx == -1) :
                        shiftx =-self.Lx
                        wcx = mx-1
                    elif (wcx==mx) :
                        shiftx = self.Lx
                        wcx = 0
                    shifty = 0.
                    #if (wcy == -1) :
                    #    shifty =-self.Ly
                    #    wcy = my-1
                    #elif (wcy==my) :
                    #    shifty = self.Ly
                    #    wcy = 0
                    shiftz = 0.
                    #if (wcz == -1) :
                    #    shiftz =-self.Lz
                    #    wcz = mz-1
                    #elif (wcz==mz) :
                    #    shiftz = self.Lz
                    #    wcz = 0
                    c1 = mz*(my*wcx+wcy)+wcz
                    #if c1 < len(self.indp) -1 :   ####### 
                    nk = self.indp[c1+1] - self.indp[c1]
                    #rc1x = append(rc1x,self.rcx[self.indp[c1]:self.indp[c1+1]] + shiftx)
                    #rc1y = append(rc1y,self.rcy[self.indp[c1]:self.indp[c1+1]] + shifty)
                    #rc1z = append(rc1z,self.rcz[self.indp[c1]:self.indp[c1+1]] + shiftz)
                    rc1x[ik:(ik+nk)] = self.rcx[self.indp[c1]:self.indp[c1+1]] + shiftx
                    rc1y[ik:(ik+nk)] = self.rcy[self.indp[c1]:self.indp[c1+1]] + shifty
                    rc1z[ik:(ik+nk)] = self.rcz[self.indp[c1]:self.indp[c1+1]] + shiftz
                    ik += nk
                    #

                    
                for i in range(self.indp[c],self.indp[c+1]):
                    #
                    # vir = 0. # to vectorize closing if after marks
                    dx = self.rcx[i] - rc1x[0:ik]  
                    dy = self.rcy[i] - rc1y[0:ik]  
                    dz = self.rcz[i] - rc1z[0:ik]  
                    r2 = dx*dx + dy*dy + dz*dz 
                    #
                    # creating a vector mask 
                    mk =  logical_and(r2 < self.r2cut, r2 > 1e-10 )
                    dx_k = dx[mk]
                    dy_k = dy[mk]
                    dz_k = dz[mk]
                    rr2 = 1./r2[mk]
                    rr6 = rr2*rr2*rr2
                    #
                    ej  = (self.c12*rr6 -self.c6)*rr6 + self.ecut
                    ene += sum(ej)
                    vir = (self.cf12*rr6-self.cf6)*rr6
                    vip += sum(vir)
                    # observable --> energy current
                    e1xx[i] = sum(ej)
                    
                    # forces
                    vir *= rr2
                    f1x[i] += sum(vir*dx_k)
                    
                    f1y[i] += sum(vir*dy_k)
                    
                    f1z[i] += sum(vir*dz_k)
                    
                    # observable --> stress tensor
                    s1xx[i] += sum(vir*dx_k*dx_k)
                    
                    s1yy[i] += sum(vir*dy_k*dy_k)
                    
                    s1zz[i] += sum(vir*dz_k*dz_k)
                    
                    s1xy[i] += sum(vir*dx_k*dy_k)
                    
                    s1xz[i] += sum(vir*dx_k*dz_k)
                    
                    s1yz[i] += sum(vir*dy_k*dz_k)
                        
            
                
    # final reordering of atomic forces , energies and stresses
    for i in range(N): 
        self.fx[self.indcp[i]]   = f1x[i]
        self.fy[self.indcp[i]]   = f1y[i]
        self.fz[self.indcp[i]]   = f1z[i]
        self.etxx[self.indcp[i]] = e1xx[i]
        self.stxx[self.indcp[i]] = s1xx[i]
        self.styy[self.indcp[i]] = s1yy[i]
        self.stzz[self.indcp[i]] = s1zz[i]
        self.stxy[self.indcp[i]] = s1xy[i]
        self.stxz[self.indcp[i]] = s1xz[i]
        self.styz[self.indcp[i]] = s1yz[i]   
        
    return ( 0.5*ene, 0.5*vip )
    
  def calcexyz(self, N) :
    # zeroing
    ewg= 0.
    fwl= 0.
    fwr= 0.
    fwt= 0.
    fwb= 0.
    # loop over particles inside selected cell
    for i in range(N):
        self.fy[i] += self.gforce
        ewg -= self.gforce*self.y[i]
        # top & bottom walls along y
        dyt = self.ywt  - self.y[i] 
        dyb = self.y[i] - self.ywb
        if(dyt < self.rwcut) :
            rr1 = 1./dyt
            rr3 = rr1*rr1*rr1
            rr6 = rr3*rr3
            ej  = (self.c9*rr6 -self.c3)*rr3 + self.ewcut
            fyt = (self.cf9*rr6 -self.cf3)*rr3*rr1
            ewg+= ej
            fwt+= fyt
            # force on particle
            self.fy[i] -= fyt
        elif(dyb < self.rwcut) :
            rr1 = 1./dyb
            rr3 = rr1*rr1*rr1
            rr6 = rr3*rr3
            ej  = (self.c9*rr6 -self.c3)*rr3 + self.ewcut
            fyb = (self.cf9*rr6 -self.cf3)*rr3*rr1
            ewg+= ej
            fwb-= fyb
            # force on particle
            self.fy[i] += fyb
        # left & right walls along z
        dzr = self.zwr  - self.z[i] 
        dzl = self.z[i] - self.zwl
        if(dzr < self.rwcut) :
            rr1 = 1./dzr
            rr3 = rr1*rr1*rr1
            rr6 = rr3*rr3
            ej  = (self.c9*rr6 -self.c3)*rr3 + self.ewcut
            fzr = (self.cf9*rr6 -self.cf3)*rr3*rr1
            ewg+= ej
            fwr+= fzr
            # forces
            self.fz[i] -= fzr
        elif(dzl < self.rwcut) :
            rr1 = 1./dzl
            rr3 = rr1*rr1*rr1
            rr6 = rr3*rr3
            ej  = (self.c9*rr6 -self.c3)*rr3 + self.ewcut
            fzl = (self.cf9*rr6 -self.cf3)*rr3*rr1
            ewg+= ej
            fwl-= fzl
            # forces
            self.fz[i] += fzl
        # contributions to atomic stress tensor (to be added)
    return( ewg, fwr, fwl, fwt, fwb)


  def eqmdi(self, N, mx, my, mz, kt, pas, mode):
    # initial evaluation of forces, energies and virials
    self.cells(mx, my, mz, N)
    (enep, virial) = self.calcener(mx, my, mz, N)
    print("# Test intra (pot.energy, virial) \n", (enep,virial) )
    (ewg, self.fwr, self.fwl, self.fwt, self.fwb)  = self.calcexyz(N)
    print("# Test pot.energy: W+G, forces: (r, l, t, b) \n", ewg, (self.fwr, self.fwl, self.fwt, self.fwb) )
    if mode == 2 : 
        # andersen thermostats: velocity sampling from maxwellian
        pstd = sqrt(self.m*kt)
        self.px[0:N] = random.normal(0., pstd, N)
        self.py[0:N] = random.normal(0., pstd, N)
        self.pz[0:N] = random.normal(0., pstd, N)
        vcmx = sum(self.px)
        vcmy = sum(self.py)
        vcmz = sum(self.pz)
        self.px[0:N]-= vcmx/N
        self.py[0:N]-= vcmy/N
        self.pz[0:N]-= vcmz/N
        print("# velocities sampled from maxwell distribution at timestep" , pas)
    vcmx = sum(self.px)
    vcmy = sum(self.py)
    vcmz = sum(self.pz)
    enek = 0.5*sum(self.px**2+self.py**2+self.pz**2)/self.m 
    self.ept = 0.
    self.ekt = 0.
    self.pres= 0.
    self.rload= 0.
    self.lload= 0.
    self.tload= 0.
    self.bload= 0.
    return (enep, enek, ewg, vcmx, vcmy, vcmz)   


  def eqmdr(self, N, mx, my, mz, kt, pas, dt, freq):
    dth=0.5*dt
    for ip in range(freq):
        pas+= 1
        t   = pas*dt
        # momenta first 
        self.px[0:N] += self.fx[0:N]*dth
        self.py[0:N] += self.fy[0:N]*dth
        self.pz[0:N] += self.fz[0:N]*dth        
        # positions second
        self.x[0:N]  += dt*self.px[0:N]/self.m
        self.y[0:N]  += dt*self.py[0:N]/self.m
        self.z[0:N]  += dt*self.pz[0:N]/self.m
        if self.mode:
            self.zwr += dt*(self.fwr-self.fwall)/self.gamma
        # compute forces
        self.cells(mx, my, mz, N)
        (enep, virial) = self.calcener( mx, my, mz, N)
        (ewg, self.fwr, self.fwl, self.fwt, self.fwb) = self.calcexyz(N)
        # momenta third
        self.px[0:N] += self.fx[0:N]*dth
        self.py[0:N] += self.fy[0:N]*dth
        self.pz[0:N] += self.fz[0:N]*dth   
        # one step advanced 
        vcmx = sum(self.px)
        vcmy = sum(self.py)
        vcmz = sum(self.pz)
        enek = 0.5*sum(self.px**2+self.py**2+self.pz**2)/self.m
        self.ekt += enek
        self.ept += enep
        self.pres+= virial       
        self.rload+= self.fwr
        self.lload+= self.fwl
        self.tload+= self.fwt
        self.bload+= self.fwb
    return (t, enep, enek, ewg, vcmx, vcmy, vcmz)     


  def nemdi(self, N, mx, my, mz, kt, dkt, pas, lther, rth2, option):
    tlef = 3.*kt
    trig = 3.*(kt + dkt)
    # initial evaluation of forces, energies and virials
    self.cells(mx, my, mz, N)
    (enep, virial) = self.calcener(mx, my, mz, N)
    print("# Test intra (pot.energy, virial) \n", (enep,virial) )
    (ewg, self.fwr, self.fwl, self.fwt, self.fwb)  = self.calcexyz(N)
    print("# Test pot.energy: W+G, forces: (r, l, t, b) \n", ewg, (self.fwr, self.fwl, self.fwt, self.fwb) )
    # initializing soret thermostats: velocity rescaling
    hista,eneka,heat,gz,jmaz,jez,ene = self.therm(tlef,trig,N,lther,rth2,option)
   #
    self.heatt[:,pas] =  (heat[:])
   #
    self.jmazt[:,pas] =  ( jmaz[:])
    self.gzt[:,pas]   =  (   gz[:])
    self.jezt[:,pas]  =  (  jez[:])
   #
    self.hstat[:,pas] =  (hista[:])
    self.enkat[:,pas] =  (eneka[:])
    self.enet[:,pas]  =  (  ene[:])
#
    vcmx = sum(self.px)
    vcmy = sum(self.py)
    vcmz = sum(self.pz)
    enek = 0.5*sum(self.px**2+self.py**2+self.pz**2)/self.m
    # initializing counters and constants
    self.ept  = 0.
    self.ekt  = 0.
    self.pres = 0.
    self.rload= 0.
    self.lload= 0.
    self.tload= 0.
    self.bload= 0.
    ## print("# starting dnemd trajectory")
    ## print( "( 'pas', 'enep', 'enek', 'enet', 'heatin', 'heatout', 'vcm' )")
    ## print( (0., enep/N, enek/N, (enep+enek)/N, heat[0], heat[1], vcmx, vcmy, vcmz) )
    return (enep, enek, ewg, vcmx, vcmy, vcmz)
    
  def nemdr(self, N, mx, my, mz, kt, dkt, pas, dt, freq, lther, rth2, option):
    tlef = 3.*kt
    trig = 3.*(kt + dkt)
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
        self.x[0:N]  += dt*self.px[0:N]/self.m
        self.y[0:N]  += dt*self.py[0:N]/self.m
        self.z[0:N]  += dt*self.pz[0:N]/self.m
        if self.mode:
            self.zwr   += dt*(self.fwr-self.fwall)/self.gamma
        # compute forces
        self.cells(mx, my, mz, N)
        (enep, virial) = self.calcener(mx, my, mz, N)
        (ewg, self.fwr, self.fwl, self.fwt, self.fwb) = self.calcexyz(N)
        # momenta third
        self.px[0:N] += self.fx[0:N]*dth
        self.py[0:N] += self.fy[0:N]*dth
        self.pz[0:N] += self.fz[0:N]*dth   
        # thermostats: velocity rescaling
        hista,eneka,heat,gz,jmaz,jez,ene = self.therm(tlef,trig,N,lther,rth2,option)
        #
        self.heatt[:,pas] =  (heat[:])
        #
        self.jmazt[:,pas] =  ( jmaz[:])
        self.gzt[:,pas]   =  (   gz[:])
        self.jezt[:,pas]  =  (  jez[:])
        #
        self.hstat[:,pas] =  (hista[:])
        self.enkat[:,pas] =  (eneka[:])
        self.enet[:,pas]  =  (  ene[:])
        #
        #end inner --> single step wrap-up
        vcmx = sum(self.px)
        vcmy = sum(self.py)
        vcmz = sum(self.pz)
        enek = 0.5*sum(self.px**2+self.py**2+self.pz**2)/self.m
        self.ekt += enek
        self.ept += enep
        self.pres+= virial   
        self.rload+= self.fwr
        self.lload+= self.fwl
        self.tload+= self.fwt
        self.bload+= self.fwb
        ## if (pas)%freq==0 : 
            ## fout.write (" %8.3f %9.4f %9.4f %10.7f %8.3f %8.3f  %7.2g %7.2g %7.2g \n" % \
    ## print( (t, enep/N, enek/N, (enep+enek)/N, heat[0], heat[1], vcmx, vcmy, vcmz) )
    return (t, enep, enek, ewg, vcmx, vcmy, vcmz)
   # end of ne-md run
    
  def therm(self,tlef,trig,N,lther,rth2,option) :
    # 1 or 2 thermostats: velocity rescaling at fixed Temperature
    # plus option with constant heating flux for 10^5 step on the right side
    flux = 1.e-5*(trig-tlef)*N   ### factor 3 already in tlef,trig
    nindl= zeros(N, dtype=int32 )
    nla  = zeros(self.lb)
    eca  = zeros(self.lb)
    vx   = zeros(self.lb)
    vy   = zeros(self.lb)
    vz   = zeros(self.lb)
    jmaz = zeros(self.lb)
    jez  = zeros(self.lb)
    e    = zeros(self.lb)
    ppx  = zeros(N)
    ppy  = zeros(N)
    ppz  = zeros(N)
    mvx  = zeros(N)
    mvy  = zeros(N)
    mvz  = zeros(N)
    heat = zeros(3)
    for i in range(N) :
        li = int((self.rz[i]    )*self.lb) 
        nla[li] += 1.
        nindl[i] = li
        vx[li]+= self.px[i]
        vy[li]+= self.py[i]
        vz[li]+= self.pz[i]
    ml   = zeros(self.lb)
    for li in range(self.lb) :
        ml[li] = (nla[li]*self.m)
    for i in range(N) :
        li = nindl[i] 
        #if(ml[li]>0.):
        mvx[i]  = self.m*vx[li]/ml[li]
        ppx[i]  = self.px[i]-mvx[i]
        mvy[i]  = self.m*vy[li]/ml[li]
        ppy[i]  = self.py[i]-mvy[i]
        mvz[i]  = self.m*vz[li]/ml[li]
        ppz[i]  = self.pz[i]-mvz[i]
        eca[li]+=(ppx[i]**2 + ppy[i]**2 + ppz[i]**2) 
    eca /=self.m
    # thermostatting lther layers on the L and R side of the box
    shift=2
    lfact= ones(self.lb)
    if option==0:
        lfact[shift:shift+lther]= tlef*(nla[shift:shift+lther]-1.)/eca[shift:shift+lther]
        heat[2]                 = sum(0.5*(lfact[shift:shift+lther]-1.)*eca[shift:shift+lther])
        eca[shift:shift+lther] *= lfact[shift:shift+lther]
        lfact[shift:shift+lther]= sqrt(lfact[shift:shift+lther])
    #
    rfact= ones(self.lb)
    lth2=rth2-lther
    rfact[lth2:rth2]= trig*(nla[lth2:rth2]-1.)/eca[lth2:rth2]    
    if option==2:
        rfact[lth2:rth2]= (eca[lth2:rth2]+flux)/eca[lth2:rth2]
    heat[0] = self.zwr
    heat[1] = sum(0.5*(rfact[lth2:rth2]-1.)*eca[lth2:rth2])
    eca[lth2:rth2] *= rfact[lth2:rth2]
    rfact[lth2:rth2]= sqrt(rfact[lth2:rth2])
    # rescale velocities of particles 
    for i in range(N) :
        li = nindl[i]
        if (li > shift) and (li < shift+lther):
            ppx[i]*= lfact[li]
            ppy[i]*= lfact[li]
            ppz[i]*= lfact[li]
            self.px[i] = ppx[i]+mvx[i]
            self.py[i] = ppy[i]+mvy[i]
            self.pz[i] = ppz[i]+mvz[i]
        elif (li >= rth2-lther) and (li < rth2) :
            ppx[i]*= rfact[li]
            ppy[i]*= rfact[li]
            ppz[i]*= rfact[li]
            self.px[i] = ppx[i]+mvx[i]
            self.py[i] = ppy[i]+mvy[i]
            self.pz[i] = ppz[i]+mvz[i]
# put back all analysis
    for i in range(N) :
        self.etxx[i] += (ppx[i]**2 + ppy[i]**2 + ppz[i]**2)/self.m
    for i in range(N) :
        li = nindl[i] 
        e[li]   += self.etxx[i]
        jmaz[li]+= ppz[i]
        jez[li] += 0.5*(ppz[i]*self.etxx[i] + self.stxz[i]*ppx[i] + self.styz[i]*ppy[i] + self.stzz[i]*ppz[i])/self.m
    #    
    return(nla, eca, heat, vz, jmaz, jez, e)

