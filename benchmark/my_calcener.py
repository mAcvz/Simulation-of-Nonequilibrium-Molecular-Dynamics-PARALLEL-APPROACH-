from numpy import zeros, rint, sqrt, sum, pi, int64, float64
from numpy import int32
import numpy 
from numba import cuda
from math import ceil
#import cupy as cp
        
def calcener( mx, my, mz, N, rx, ry, rz, Lx, Ly, Lz, sp, r2cut, c12, c6, ecut, cf12, cf6, fx, fy, fz, etxx, stxx, styy, stzz, stxy, stxz, styz) :
    # zeroing
    # nb: rectangular PBC
    ene=0.
    vip=0.
    # arrey (numpy) temporanei 
    t_e1xx = zeros(N) 
    t_s1xx = zeros(N)
    t_s1yy = zeros(N)
    t_s1zz = zeros(N)
    t_s1xy = zeros(N)
    t_s1xz = zeros(N)
    t_s1yz = zeros(N)
    t_f1x  = zeros(N)
    t_f1y  = zeros(N)
    t_f1z  = zeros(N)
    # posizioni  assolute delle particelle (da ricavare partendo da rx,ry,rz)
    rcx = zeros(N)
    rcy = zeros(N)
    rcz = zeros(N)
    # numero di celle in cui è stata divisa la box
    ncells=mx*my*mz
    # np conterrà il numero di particelle nella cella considerata
    np   = zeros(ncells, dtype=int32)
    # l'indice 'indc' contiene l'indice della cella in cui si trova l'i-esima particella (mi dirà dove si trova ogni particella)
    indc = zeros(N, dtype=int32)
    # tiene traccia della specie atomica 
    s1p  = zeros(N, dtype=int32)
    # generare l'indice della cella c-esima
    for i in range(N):
        vcx=int(mx*(rx[i]+0.5))
        vcy=int(my*(ry[i]+0.5)) 
        vcz=int(mz*(rz[i]+0.5))
        # cell index
        c = mz*(my*vcx+vcy)+vcz
        # ad ogni particella associo la sua cella di appartenenza
        indc[i]=c
        # vado ad accumulare il numero di particelle dentro ongi cella c 
        np[c] += 1
    # contiene il numero di particelle contenute nelle prime n celle ( con 0 <= n <= ncells)
    indp = zeros(ncells+1, dtype=int32)
    for c in range(0,ncells) :
        indp[c+1] = indp[c] + np[c]
    # vettore che servirà a riordinare le particelle 
    indcp= zeros(N, dtype=int32)
    for i in range(N):
        '''
            la coppia (i,c) rapresenta la i-esima particella e la sua cella di
            appartenenza. 
            Si usa indp[c] += 1 in modo da visitare tutte le particelle conenute
            nella c-esima cella. 
            ## rx,ry,rz sono le posizioni relative definite come segue:
                self.rx  =  self.x/self.Lx
                self.rx -= rint(self.rx)
        '''
        c=indc[i] # c è l'indice cella di appartenenza della particella i-esima considerata
        # riottengo la posizione rale delle particelle. 
        rcx[indp[c]] = (rx[i]+0.5)*Lx #  0 < rcx < Lx
        rcy[indp[c]] = (ry[i]+0.5)*Ly #  0 < rcy < Ly
        rcz[indp[c]] = (rz[i]+0.5)*Lz #  0 < rcz < Lz
        s1p[indp[c]] = sp[i] # specie atomica della partcielle i-esima
        indcp[indp[c]] = i
        indp[c] += 1 # aggiungo 1 in modo da visitare tutte le particelle contenute nella cella c-esima 
    #
    # need to reconstruct index list
    # infatti per visitare tutte le particelle abbiamo eseguito indp[c] += 1 e abbiamo alterato indp
    indp[0]=0
    for c in range(0,ncells) :
        indp[c+1] = indp[c] + np[c]
    # indexing neighbour cells of selected one
    #
    ncmax = int(numpy.amax(np)) # numero massimo di particelle contenuto in un sola cella 
    nter = int(ncmax*27) # stima per eccesso il numero di particelle contenuto nelle 27 celle vicine 
    vcx1 = zeros(27, dtype=int32) # vettore di numeri interi usati per riscostruire l'indice c1 della cella 
    vcy1 = zeros(27, dtype=int32) # vettore di numeri interi usati per riscostruire l'indice c1 della cella 
    vcz1 = zeros(27, dtype=int32) # vettore di numeri interi usati per riscostruire l'indice c1 della cella 
    #
    rci = zeros((ncmax,mz,3)) # permette di tenere traccia della posizione (x,y,z) di ogni particella in una particoalre cella "c" lungo la direzone z 
    rcn = zeros((nter,mz,3))  # permette di tenere traccia della posizione (x,y,z) di ogni particella in una delle 27 celle (lungo la direzone z) 
    s1pi = zeros((ncmax,mz)) # inerente alla specie atomica 
    s1pn = zeros((nter,mz)) # inerente alla specie atomica 
    #
    npi = zeros(mz,dtype=int32) #  per ogni cella lungo z conta il numero di particelle contenute
    npj = zeros(mz,dtype=int32) #  tengo traccia del numero di particelle contenute nelle 27 celle prime vicine 
    # vcx,vcy,vcz: interi necessari a calcolare la posizioni delle celle vicine nel cubo 3x3
    k=0
    #i.j,l variano tra -1,0,1: posso visitare tutto il cubo 3x3 centrato in (0,0,0)
    for i in range(-1,2):
        for j in range(-1,2):
            for l in range(-1,2):
                vcx1[k]= i
                vcy1[k]= j
                vcz1[k]= l
                k+=1
    #
    # Loop over Cells
    # vcx,vcy,vcz sono gli interi necessari per ricostruire l'indice della cella 
    for vcx in range(mx):
        for vcy in range(my):
            # arrays temporanei  che conterrano le grandezze volute relative alle particelle di una cella lungo z
            # ncmax corrisponde al numero massimo di particelle contenute in un singaola cella.
            c_e1xx = zeros((ncmax,mz))
            c_s1xx = zeros((ncmax,mz))
            c_s1yy = zeros((ncmax,mz))
            c_s1zz = zeros((ncmax,mz))
            c_s1xy = zeros((ncmax,mz))
            c_s1xz = zeros((ncmax,mz))
            c_s1yz = zeros((ncmax,mz))
            c_f1x  = zeros((ncmax,mz))
            c_f1y  = zeros((ncmax,mz))
            c_f1z  = zeros((ncmax,mz))
            # scorro le celle lungo la direzione z     
            for vcz in range(mz):
                # creo l'indice della cella c-esima
                c = mz*(my*vcx+vcy)+vcz
                # loop over particles inside selected cells (central+neighbours)
                conti = 0
                for i in range(indp[c],indp[c+1]):
                    # vado ad immagazzinare le posizioni delle particelle nella cella c 
                    rci[conti,vcz,0] = rcx[i]
                    rci[conti,vcz,1] = rcy[i]
                    rci[conti,vcz,2] = rcz[i]
                    s1pi[conti,vcz] = s1p[i] # tengo traccia del tipo di particella nella cella "c"
                    conti += 1
                npi[vcz] = conti  # tengo traccia del numero di particelle contenute nella cella "c"
                contj = 0
                # indice "c1" di una delle 27 celle vicine (lo oeetengo a partire da 'c' e poi mi spsoto nelle 3 direzioni di +/- 1, 0)
                for k in range(27) :
                    wcx=vcx + vcx1[k]
                    wcy=vcy + vcy1[k]
                    wcz=vcz + vcz1[k]
                    # Periodic boundary conditions 
                    shiftx = 0.
                    if (wcx == -1) :
                        shiftx =-Lx
                        wcx = mx-1
                    elif (wcx==mx) :
                        shiftx = Lx
                        wcx = 0
                    shifty = 0.
                    if (wcy == -1) :
                        shifty =-Ly
                        wcy = my-1
                    elif (wcy==my) :
                        shifty = Ly
                        wcy = 0
                    shiftz = 0.
                    if (wcz == -1) :
                        shiftz =-Lz
                        wcz = mz-1
                    elif (wcz==mz) :
                        shiftz = Lz
                        wcz = 0
                    # determino l'indice della cella prima viciana c1
                    c1 = mz*(my*wcx+wcy)+wcz
                    # eseguimao un loop sulle particelle della cella c1 e salvo le posizioni delle particelle in "c1"
                    for j in range(indp[c1],indp[c1+1]):
                        rcn[contj,vcz,0] = rcx[j]+shiftx
                        rcn[contj,vcz,1] = rcy[j]+shifty
                        rcn[contj,vcz,2] = rcz[j]+shiftz
                        s1pn[contj,vcz] = s1p[j] # tengo traccia del tipo di particella nella cella "c1"
                        contj += 1
                # il ciclo è stato ripetuto 27 volte quindi contj = numero di particelle nelle 27 celle 
                npj[vcz] = contj # tengo traccia del numero di particelle contenute nelle celle prime vicine a c
                

            # CHIAMATA KERNEL
            threadsperblock = (16,16)
            # mi assicuro che ci sia un numero interro di thrads 
            blockspergrid_x = ceil(rci.shape[0] / threadsperblock[0])
            blockspergrid_y = ceil(npj.shape[0] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            calgpu[blockspergrid, threadsperblock](rci,rcn,npi,npj,r2cut,s1pi,s1pn,c12,c6,cf12,cf6,ecut,c_f1x,c_f1y,c_f1z,c_s1xx,c_s1yy,c_s1zz,c_s1xy,c_s1xz,c_s1yz,c_e1xx)
            #
            '''
            # call back on cpu 
            c_e1xx = cp.asnumpy(g_e1xx) 
            c_s1xx = cp.asnumpy(g_s1xx)
            c_s1yy = cp.asnumpy(g_s1yy)
            c_s1zz = cp.asnumpy(g_s1zz)
            c_s1xy = cp.asnumpy(g_s1xy)
            c_s1xz = cp.asnumpy(g_s1xz)
            c_s1yz = cp.asnumpy(g_s1yz)
            c_f1x  = cp.asnumpy(g_f1x )
            c_f1y  = cp.asnumpy(g_f1y )
            c_f1z  = cp.asnumpy(g_f1z )
            '''
            # trasferire  su array temporanei che contengano tutte le N particelle
            for vcz in range(mz) :
                # genero nuovamente l'indice della cella 
                c = mz*(my*vcx+vcy)+vcz
                for i in range(npi[vcz]):
                    ii =  indp[c] + i
                    t_f1x[ii]  = c_f1x[i,vcz]
                    t_f1y[ii]  = c_f1y[i,vcz]
                    t_f1z[ii]  = c_f1z[i,vcz]
                    t_e1xx[ii] = c_e1xx[i,vcz]
                    t_s1xx[ii] = c_s1xx[i,vcz]
                    t_s1yy[ii] = c_s1yy[i,vcz]
                    t_s1zz[ii] = c_s1zz[i,vcz]
                    t_s1xy[ii] = c_s1xy[i,vcz]
                    t_s1xz[ii] = c_s1xz[i,vcz]
                    t_s1yz[ii] = c_s1yz[i,vcz]
  
    # final reordering of atomic forces , energies and stresses
    for i in range(N): 
        fx[indcp[i]]   = t_f1x[i]
        fy[indcp[i]]   = t_f1y[i]
        fz[indcp[i]]   = t_f1z[i]
        etxx[indcp[i]] = t_e1xx[i]
        ene += t_e1xx[i]
        stxx[indcp[i]] = t_s1xx[i]
        styy[indcp[i]] = t_s1yy[i]
        stzz[indcp[i]] = t_s1zz[i]
        vip += t_s1xx[i]+t_s1yy[i]+t_s1zz[i]
        stxy[indcp[i]] = t_s1xy[i]
        stxz[indcp[i]] = t_s1xz[i]
        styz[indcp[i]] = t_s1yz[i]    
    
    
    return ( 0.5*ene, 0.5*vip )

@cuda.jit
def calgpu(rci,rcn,npi,npj,r2cut,s1pi,ns1p,c12,c6,cf12,cf6,ecut,f1x,f1y,f1z,s1xx,s1yy,s1zz,s1xy,s1xz,s1yz,e1xx):
    # 
    # rci = zeros((ncmax,mz,3)) # permette di tenere traccia della posizione (x,y,z) di ogni particella in una particoalre cella c lungo la direzone z 
    # rcn = zeros((nter,mz,3))  # permette di tenere traccia della posizione (x,y,z) di ogni particella in una particoalre cella c1(3x3) lungo la direzone z 
    # s1pi = zeros((nter,mz)) # inerente alla specie atomica 
    # s1pn = zeros((nter,mz)) # inerente alla specie atomica 
    # npi = zeros(mz) # npi per ogni cella lungo z conta il numero di particelle contenute
    # npj = zeros(mz) # conta il numero di particelle contenute in tutte 27 le celle vicine 
    #
    # pos 1 scorre sulle particelle  all'interno di una cella 
    # pos 2 scorre sulle celle lungo z con x,y fissate
    pos1,pos2 = cuda.grid(2)
    if pos2 < npj.shape[0]:  # pos2 deve sempre essere minore di mz ovvero npj.shape[0]
        if pos1 < npi[pos2]: # pos1 deve sempre essere minore del numero di particelle contenute nella cella considerata
            # scorro sulle particelle nelle celle vicine (una particoalre "pos2")
            for j in range(npj[pos2]) :
                dx = rci[pos1,pos2,0] - rcn[j,pos2,0] 
                dy = rci[pos1,pos2,1] - rcn[j,pos2,1]
                dz = rci[pos1,pos2,2] - rcn[j,pos2,2]
                r2 = dx*dx + dy*dy + dz*dz
                if (r2<r2cut and r2>0.1) :
                    rr2 = 1./r2
                    rr6 = rr2*rr2*rr2
                    index = int(s1pi[pos1,pos2] + ns1p[j,pos2])
                    eij = (c12[index]*rr6 - c6[index])*rr6 + ecut[index]
                    vij = (cf12[index]*rr6 - cf6[index])*rr6 
                    vij *= rr2
                    
                    e1xx[pos1,pos2] += eij
                    f1x[pos1,pos2]  += vij * dx
                    f1y[pos1,pos2]  += vij * dy
                    f1z[pos1,pos2]  += vij * dz 
                    s1xx[pos1,pos2] += vij * dx * dx
                    s1yy[pos1,pos2] += vij * dy * dy 
                    s1zz[pos1,pos2] += vij * dz * dz 
                    s1xy[pos1,pos2] += vij * dx * dy 
                    s1xz[pos1,pos2] += vij * dx * dz
                    s1yz[pos1,pos2] += vij * dy * dz
                    
                    '''
                        al termine dell'esecuzione di calgpu ho le grandezze di inteserre 
                        definite su tutte le particelle contenute su tutte le celle lungo z. 
                    '''
                
