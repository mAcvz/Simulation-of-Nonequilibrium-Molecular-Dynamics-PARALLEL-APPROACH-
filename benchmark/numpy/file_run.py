from numpy import zeros, pi, sin, cos, empty, array, float64
from m_lj import LJ
from m_write import *
from time import process_time
from  pickle import dump, load, HIGHEST_PROTOCOL

if __name__ == "__main__":
    # r-parameters # defaults
    dir="./RUN/"
    nstep= 10
    rho  = 0.984   #  Xe 0.634
    mx   = 4       #  8 Xe 5
    my   = 4       #  10 Xe 5
    mz   = 4       #  25 Xe 16
    zbuf = 4       #
    kt   = 0.79    # Xe 1.115
    dk0  = 0.00    # equilibrio
    dkt  = 0.10    # 0.20 Xe 0.11
    dt   = 0.005   # 0.005
    freq = 1       #  2000
    lther= 3       #  3 ( shifted by 2)
    gacc = 0.      # gravity acceleration (force with m=1)
    fwall= 1300.   # fixing lateral pressure
    rth2 = 2*(mz+1)#  lb=2*mz - shift
    my  += 2       # 
    mz  += zbuf    # 
    print( "# number of boxes mx = %d, my = %d, mz = %d" % (mx, my, mz) )
    print( "# mean (kinetic) temperature kt = %8.4f " % (kt) )
    print( "# integration time step dt = %8.4f" % dt )
    print( "# number of time steps for this run nstep = %d" % (nstep) )    
    t0 = process_time()
    IN = LJ(rho, gacc, mx, my, mz, zbuf, nstep*freq)
    IN.mode = 0
    IN.fwall = fwall
    print( "# creating EQ object, cpu = %8.4f" % (process_time()-t0) )
    print( "# density rho = %8.4f  (=%8.4f)" % (IN.rho, rho))
    print( "# temperature T = %8.4f  (dT=%8.4f)" % (kt, dkt))
    print( "# external conditions: gravity g=%8.4f" % gacc )
    print( "# external conditions: right wall force  Fw=%9.4f " % fwall )
    N = IN.npart
    rhofact = IN.lb / (IN.Lx * IN.Ly * IN.Lz)
    print( "# sides of the rectangular MD-box L = [ %8.4f %8.4f %8.4f ]" % (IN.Lx, IN.Ly, IN.Lz) )
    print( "# number of particles  N = %d" % N)
    print( "# potential cut-off radius  rcut = %8.4f *sigma" % IN.rcut)
    print( "# lateral size of slicing Deltaz = %8.4f" % (IN.Lz / (IN.lb)))
    #
    #
    # 
    # starting from (fcc) lattice
    IN.fcc()
    writexyz(IN, N, filexyz=dir+'fcc.xyz')
    #
    #
    #
    #
    pas = 0
    mode=2
    t0 = process_time()    
    (enep, enek, ewg, vcmx, vcmy, vcmz) = IN.eqmdi( N, mx, my, mz, kt, pas, mode)
    print( "# initial conditions step = %d, cpu = %8.4f" % (pas, process_time()-t0) )
    print( "     'pas'    'enep'    'enek'       'enet' 'virial'    'fwl'    'fwr'    'zwr'       'vcm' ")
    etot = enep+enek+ewg
    print(" %9.2f %9.4f %9.4f %12.3f %8.4f %8.3f %8.2f %8.2f %7.1g %7.2g %7.2g" % (pas*dt,enep/N, enek/N, etot, ewg, IN.fwl, IN.fwr, IN.zwr, vcmx, vcmy, vcmz) )
    write_input(IN, N, nstep, conf_out=dir+"conf_fcc.b")
    #
    #
    #
    # Equilibrium run with temperature from maxwellian sampling
    print("# starting eqmd trajectory\n")
    print( "     'pas'    'enep'    'enek'       'enet'    'ewg'    'fwl'    'fwr'    'zwr'       'vcm' ")
    #etot = enep+enek
    nstep= 10    
    t0 = process_time()
    for it in range(nstep):
        (t, enep, enek, ewg, vcmx, vcmy, vcmz) = IN.eqmdr( N, mx, my, mz, kt, pas, dt, freq)
        etot = enep+enek+ewg
        pas += freq
        print(" %9.2f %9.4f %9.4f %12.3f %8.3f %8.2f %8.2f %8.3f %7.1g %7.2f %7.2f" % (pas*dt,enep/N, enek/N, etot, ewg, IN.fwl, IN.fwr, IN.zwr, vcmx, vcmy, vcmz) )
    print( "# (IN)-Equilibrium run, cpu = %8.4f" % (process_time()-t0) )
    g=3*N-1
    nstep *= freq
    print( "# ending initial equilibrium run  <ep>=%6.1f  <ek>=%7.4f   T=%8.3f  P=%8.3f\n" % ( IN.ept/nstep, IN.ekt/nstep, 2.*IN.ekt/(nstep*g), IN.pres/nstep) )
    print( "# ending initial equilibrium run  <Lload>=%8.2f  <Rload>=%8.2f   " % ( IN.lload/nstep, IN.rload/nstep) )
    print( "# ending initial equilibrium run  <Bload>=%8.2f  <Tload>=%8.2f \n" % ( IN.bload/nstep, IN.tload/nstep) )
    # final positions and momenta to file conf_in.b
    write_input(IN, N, nstep, conf_out=dir+"conf_eq.b")
    writexyz(IN, N, filexyz=dir+'conf1.xyz')
    
