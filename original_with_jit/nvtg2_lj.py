from numba.experimental import jitclass
from numba import int32,int64,float64
from numpy import zeros, rint, sqrt, sum, pi
from numpy import random

spec = [
    ('lb', int64),
    ('dslice', int64),
    ('l1', int64),
    ('l2', int64),
    ('l3', int64),
    ('rho', float64), 
    ('npart', int64),
    ('ma', float64), 
    ('mb', float64), 
    ('gforce', float64),
    ('c6', float64[:]), 
    ('c12', float64[:]), 
    ('cf6', float64[:]), 
    ('cf12', float64[:]), 
    ('ecut', float64[:]), 
    ('rcut', float64), 
    ('Lx', float64),
    ('Ly', float64),
    ('Lz', float64),
    ('m', float64[:]),
    ('x', float64[:]),
    ('y', float64[:]),
    ('z', float64[:]),
    ('rx', float64[:]),
    ('ry', float64[:]),
    ('rz', float64[:]),
    ('sp', int64[:]),
    ('px', float64[:]),
    ('py', float64[:]),
    ('pz', float64[:]),
    ('fx', float64[:]),
    ('fy', float64[:]),
    ('fz', float64[:]),
    ('etxx', float64[:]),
    ('stxx', float64[:]),
    ('styy', float64[:]),
    ('stzz', float64[:]),
    ('stxy', float64[:]),
    ('stxz', float64[:]),
    ('styz', float64[:]),
    ('ekin', float64),
    ('ene', float64),
    ('etot', float64),
    ('ept', float64),
    ('ekt', float64),
    ('pres', float64),
    ('kg', int64),
    ('gcount', float64[:,:]),
    ('r2max', float64),
    ('r2cut', float64),
    ('ldel', float64),
    ('tt', float64),
    ('heatt', float64[:,:]),
    ('hstat', float64[:,:]),
    ('hstbt', float64[:,:]),
    ('enkat', float64[:,:]),
    ('enkbt', float64[:,:]),
    ('jmazt', float64[:,:]),
    ('gzt', float64[:,:]),
    ('jezt', float64[:,:]),
    ('enet', float64[:,:])
]

#from numpy import int32,int64,float64
@jitclass(spec)
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
    a = (4/rho)**(1./3.) # passo reticolare 
    self.dslice = 2  
    cell_lenght = self.dslice*a    # doppio del passo reticolare (della cella minima)
    self.rcut = 3.*sigb            # distanza dopo  la quale il potenziale si azzera 
    if (cell_lenght < self.rcut):  # ci assicuriamo che l'interazione si a corto raggio 
        self.rcut = cell_lenght
    '''
        suppongo che l1,l2,l3 siano il numro di siti reticolari nelle varie dimesioni 
    '''
    self.lb  =  mz*self.dslice
    self.l1  =  mx*self.dslice
    self.l2  = (my-2)*self.dslice  # assume it is enough to have one lower and one higher extra box
    self.l3  =  mz*self.dslice
    self.npart = 4*self.l1*self.l2*self.l3 # 4 particelle per ogni sito reticolare 
    self.Lx = mx * cell_lenght   # lunghezza assoluta della scataola lungo x 
    self.Ly = my * cell_lenght   # ...
    self.Lz = mz * cell_lenght   # ....
    self.rho = self.npart/(mx*(my-2)*mz*cell_lenght**3)   # densità di particelle  
    ndim = self.npart 
    # N initial guess of average number of particles for dimensioning
    # ricorda che le posizioni iniziali sono x,y,z e sono date dalla distribuzione gaussiana 
    self.x       = zeros( ndim )
    self.y       = zeros( ndim )
    self.z       = zeros( ndim )
    self.m       = zeros( ndim )
    self.sp      = zeros( ndim, dtype=int64 )
    # cordinate relative alla scatola di simulazione di dimensioni unitaria 
    self.rx      = zeros( ndim )
    self.ry      = zeros( ndim )
    self.rz      = zeros( ndim )
    # momenti 
    self.px      = zeros( ndim )
    self.py      = zeros( ndim )
    self.pz      = zeros( ndim )
    # forze 
    self.fx      = zeros( ndim )
    self.fy      = zeros( ndim )
    self.fz      = zeros( ndim )
    # tensore degli sforzi
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
    self.gcount  = zeros( (self.kg,3) ) #shape = (512,3)
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
    # particle species: a     # sta definendo le due specie atomiche e le loro masse 
    self.sp[0:Na]= 0
    self.m[0:Na] = self.ma
    # particle species: b
    self.sp[Na:N]= 1
    self.m[Na:N] = self.mb
    '''
        definiamo ele costanti di interazione Lj tra a-a a-b b-b
    '''
    # a-a interaction       
    sig6        = siga**6   # sigma alla sesta
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
    # numero di siti reticolari nelle 3 direzioni
    ax = self.Lx/self.l1
    ay = self.Ly/(self.l2 + 2*self.dslice) 
    #ay = self.Ly/self.l2 
    az = self.Lz/self.l3
    print( "# lattice lenghts (ax,ay,az) =", (ax, ay, az) )
    print( "# (mx, my, mz) =", (self.l1,self.l2, self.l3) )
    mm = self.l1*self.l2*self.l3    # numero totale di siti reticolari
    natom = 4*mm                    # numero totale di atomi -- 4 per sito 
    print( "# number of lattice cells =", mm )
    print( "# number of particles =" , natom )
    print( "# md-box sides [Lx, Ly, Lz ]=", (self.Lx, self.Ly, self.Lz) )
    j  = 0
    xi = 0.
    yi = 0.86
    zi = 0.
    delta=0.005 # deviazione standard 
    '''
        generiamo le distribuzioni gaussiane da usare per assegnare le x,y,z iniziali 
    '''
    rrx = random.normal(0., delta, natom)
    rry = random.normal(0., delta, natom)
    rrz = random.normal(0., delta, natom)
    #with open("fcc.txt", "w") as f:
    for nx in range(self.l1) :                   # stiamo scorrendo su ogni siito reticolare
        for ny in range(self.l2) :
            for nz in range(self.l3) :           # posizioniamo gli atomi della base 
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


  def calcener(self, mx, my, mz, N ) :
    # zeroing
    # nb: rectangular PBC
    ene=0.
    vip=0.
    # posizioni assolute all'interno della cella  
    rcx = zeros(N)
    rcy = zeros(N)
    rcz = zeros(N)
    # componenti delle forze risultanti 
    f1x= zeros(N)
    f1y= zeros(N)
    f1z= zeros(N)
    # tensore degli sforzi 
    e1xx = zeros(N)
    s1xx = zeros(N)
    s1yy = zeros(N)
    s1zz = zeros(N)
    s1xy = zeros(N)
    s1xz = zeros(N)
    s1yz = zeros(N)
    # numero di celle in cui è stata divisa la box
    ncells=mx*my*mz
    # np conterrà il numero di particelle nella cella della particella considerata
    np   = zeros(ncells, dtype=int32)
    # l'indice indc contiene l'indice della cella in cui si trova l'i-esimo atomo - mi dirà dove si trova ogni atomo 
    indc = zeros(N, dtype=int32) 
    s1p  = zeros(N, dtype=int32) #? forse ha a che fare con le specie atomiche
    for i in range(N):
        ''' 
            aggiung + 0.5 cosí sono sicuro di prendere l'intero corrispondente alla cella giusta,
            infatti quando vado a calcolare rx -= rint(self.rx) quindi in alcuni casi ho rx < 0.5 
            perchè si riferisce alla particella immagine e non a quella nella cella della particella considerata
        '''
        vcx=int(mx*(self.rx[i]+0.5))  # ricorda che in quasto casa si arrotonda solo per difetto 
        vcy=int(my*(self.ry[i]    )) # int(my*(self.ry[i]+0.5)) #
        vcz=int(mz*(self.rz[i]+0.5))
        # cell index
        c = mz*(my*vcx+vcy)+vcz ################# attento manca mx quindi dimensionalmente torna il conto 
        indc[i]=c # ad ogni atomo associo la sua cella di appertenenza
        np[c] += 1
    '''
        il vettore indp contiente nelle sue componenti la somma del numero di particelle contenute nelle prime i box
    '''
    indp = zeros(ncells+1, dtype=int32)  
    for c in range(0,ncells) :
        indp[c+1] = indp[c] + np[c]
    #
    indcp= zeros(N, dtype=int32)
    for i in range(N):
        c=indc[i] 
        '''
        in questo ciclo la coppia i,c è l'iesima particella e la cella di appartenenza
        # ovvimente c si ripeterà perchè piú atomi appartengono alla stessa box 
        tuttavia noi usiamo indp[c]+1 per andare a visitare tutte le particelle conenute nella stessa box
        # andiamo a definire le posizoni assolute all'interno della scatola
        # rx,ry,rz sono le posizioni relative rx = x/Lx ....
        '''
        rcx[indp[c]] = (self.rx[i]+0.5)*self.Lx #  self.rx[i]     *self.Lx #
        rcy[indp[c]] =  self.ry[i]     *self.Ly # (self.ry[i]+0.5)*self.Ly #
        rcz[indp[c]] = (self.rz[i]+0.5)*self.Lz #  self.rz[i]     *self.Lz #
        s1p[indp[c]] = self.sp[i]  # tengo in memoria la specie atomicata
        indcp[indp[c]] = i # forse contiene gli indici degli atomi però ordinati in modo da avere atomi vicini con indici vicini.
        indp[c] += 1
    # need to reconstruct index list
    # infatti per visitare tutte le particelle abbiamo eseguito indp[c] += 1 e abbiamo distrorto indp
    indp[0]=0
    for c in range(0,ncells) :
        indp[c+1] = indp[c] + np[c]
    # indexing neighbour cells of selected one
    # i vettori definiti di seguito permettono fissata un cella di raggiungere le 26 vicine
    # infatti i vettori vcx1,vcy1,vcz1 permettono di accedere al blocco 3x3 di celle 
    k = 0
    vcx1 = zeros(27, dtype=int32)
    vcy1 = zeros(27, dtype=int32)
    vcz1 = zeros(27, dtype=int32)
    for i in range(-1,2):
        for j in range(-1,2):
            for l in range(-1,2):
                vcx1[k]= i
                vcy1[k]= j
                vcz1[k]= l
                k+=1
    # Loop over Cells
    for vcx in range(mx):
        for vcy in range(my):
            for vcz in range(mz):
                c = mz*(my*vcx+vcy)+vcz  
                # loop over particles in neighbour cells 
                # wcx,wcy,wcz cotengono le cordinate per individuare le celle vicine alla c-esima
                for k in range(27) :
                    wcx=vcx + vcx1[k]
                    wcy=vcy + vcy1[k]
                    wcz=vcz + vcz1[k]
                    # Periodic boundary conditions
                    shiftx = 0.
                    if (wcx == -1) :
                        shiftx =-self.Lx  # attenzione allo shift -> nella formula compare un meno
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
                    if (wcz == -1) :
                        shiftz =-self.Lz
                        wcz = mz-1
                    elif (wcz==mz) :
                        shiftz = self.Lz
                        wcz = 0
                    c1 = mz*(my*wcx+wcy)+wcz   # è l'indeice della box considerata 
                    '''' ricorda ora rcx,rcy,rcz sono ordinate '''
                    for i in range(indp[c],indp[c+1]): # il range comprende tutte le particelle che ci sono di differenza tra la box c+1 e c 
                        for j in range(indp[c1],indp[c1+1]):  # comprende tutte le particelle che ci sono di differenza tra la box c1+1 e c1 
                            dx = rcx[i] - (rcx[j] + shiftx)  
                            dy = rcy[i] - (rcy[j] + shifty)
                            dz = rcz[i] - (rcz[j] + shiftz)
                            r2 = dx*dx + dy*dy + dz*dz  # modulo quadro della delal distanza
                            if(r2<self.r2cut and r2>0.1) :    # ci assicuriamo che sia minore di r2cut e che le particelle non siano compenetrate
                                #self.gcount[int(r2/self.ldel2)]+=2. 
                                rr2 = 1./r2
                                rr6 = rr2*rr2*rr2
                                index=s1p[i]+s1p[j]
                                #
                                ej  = (self.c12[index]*rr6 -self.c6[index])*rr6 + self.ecut[index] # LJ tenendo conto di interazioni a-a,a-b,b-b
                                ene+= ej                                                           # ricorda che (Lj-Lj_cut) per raccordare a 0
                                # observable --> energy current
                                e1xx[i]+=ej
                                #
                                vir = (self.cf12[index]*rr6-self.cf6[index])*rr6
                                vip+= vir
                                vir*=rr2
                                # forces
                                f1x[i]+=vir*dx # conta che adesso  vir = vir/r2 => la formula della forza è giusta 
                                f1y[i]+=vir*dy
                                f1z[i]+=vir*dz
                                # observable --> stress tensor
                                s1xx[i]+=vir*dx*dx
                                s1yy[i]+=vir*dy*dy
                                s1zz[i]+=vir*dz*dz
                                s1xy[i]+=vir*dx*dy
                                s1xz[i]+=vir*dx*dz
                                s1yz[i]+=vir*dy*dz
    # final reordering of atomic forces , energies and stresses
    for i in range(N): 
        self.fx[indcp[i]]   = f1x[i]
        self.fy[indcp[i]]   = f1y[i]
        self.fz[indcp[i]]   = f1z[i]
        self.etxx[indcp[i]] = e1xx[i]
        self.stxx[indcp[i]] = s1xx[i]
        self.styy[indcp[i]] = s1yy[i]
        self.stzz[indcp[i]] = s1zz[i]
        self.stxy[indcp[i]] = s1xy[i]
        self.stxz[indcp[i]] = s1xz[i]
        self.styz[indcp[i]] = s1yz[i]   

    return ( 0.5*ene, 0.5*vip ) # le divide per 2 forse perchè sono state contate due volte per ogni forza 
    
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
    return( enex, viex)

  def eqmdi(self, N, Na, mx, my, mz, kt, pas, mode):
    '''
        vado a determinare le posizioni delle particelle immagini 
    '''   
    # initial evaluation of forces, energies and virials
    self.rx  =  self.x/self.Lx
    self.rx -= rint(self.rx)
    self.ry  =  self.y/self.Ly
    #self.ry -= rint(self.ry)
    self.rz  =  self.z/self.Lz
    self.rz -= rint(self.rz )
    (enep, virial) = self.calcener(mx, my, mz, N)
    print("test 1",enep,virial)
    (enex, virex)  = self.calcexy(N)
    print("test 2",enex,virex)
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
        #self.ry[0:N]-= rint(self.ry[0:N])
        self.rz[0:N] =  self.z[0:N]/self.Lz
        self.rz[0:N]-= rint(self.rz[0:N])
        (enep, virial) = self.calcener( mx, my, mz, N)
        (enex, virex)  = self.calcexy(N)
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
    #self.ry -= rint(self.ry)
    self.rz  =  self.z/self.Lz
    self.rz -= rint(self.rz )
    (enep, virial) = self.calcener(mx, my, mz, N)
    (enex, virex)  = self.calcexy(N)
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
        self.rx[0:N] =  self.x[0:N]/self.Lx
        self.rx[0:N]-= rint(self.rx[0:N])
        self.ry[0:N] =  self.y[0:N]/self.Ly
        #self.ry[0:N]-= rint(self.ry[0:N])
        self.rz[0:N] =  self.z[0:N]/self.Lz
        self.rz[0:N]-= rint(self.rz[0:N])
        (enep, virial) = self.calcener(mx, my, mz, N)
        (enex, virex)  = self.calcexy(N)
        # momenta third
        self.px[0:N] += self.fx[0:N]*dth
        self.py[0:N] += self.fy[0:N]*dth
        self.pz[0:N] += self.fz[0:N]*dth   
        # one step advanced 
        # soret thermostats: velocity rescaling
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
    return(nla, nlb ,eca, ecb, heat, vz, jmaz, jez, e)
