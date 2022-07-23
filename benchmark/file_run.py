from numpy import zeros, pi, sin, cos, empty, array, float64
#from nvtg_lj import LJ 
#from nvtg_write import *
from time import process_time
from  pickle import dump, load, HIGHEST_PROTOCOL

# from numba import jit
if __name__ == "__main__":
    dir="./RUN/"
# r-parameters # defaults
    rho  = 0.984   #  density of the solid phase
    mx   = 6       #  number of cells along x
    my   = 6 + 2   #  number of cells along y
    mz   = 12+ 2   #  number of cells along z
    kt   = 0.7     #  temperature
    dk0  = 0.00    #  delta T at equilibrium
    dkt  = 0.70    #  delta T for melting region
    dt   = 0.005   #  timestep (LJ units)
    gforce = -0.1  #  vaule of gravity field
    freq =  2      #  printing frequency
    lther=  3      #  width of thermostat region
    nstep= 10      #  number of multiples of 'freq' time steps 
    pfile = dir + "fccmd.pickle"
    print( "# number of boxes mx = %d, my = %d, mz = %d" % (mx, my, mz) )
    print( "# mean (kinetic) temperature kt = %8.4f " % (kt) )
    print( "# integration time step dt = %8.4f" % dt )
    print( "# number of time steps for this run nstep = %d" % (nstep) )    
    t0 = process_time()
    IN = LJ(rho, mx, my, mz, gforce, nstep*freq)
    print("# creating EQ object, cpu = %8.4f" % (process_time()-t0) )
    N = IN.npart
    Na = 3*N // 25
    rhofact = IN.lb / (IN.Lx * IN.Ly * IN.Lz)
    print( "# sides of the rectangular MD-box L = [ %8.4f %8.4f %8.4f ]" % (IN.Lx, IN.Ly, IN.Lz) )
    print( "# number of particles  N = %d" % N)
    print( "# density rho = %8.4f   (%8.4f)" % (IN.rho, rho))
    print( "# potential cut-off radius  rcut = %8.4f *sigma" % IN.rcut)
    print( "# lateral size of slicing Deltaz = %8.4f" % (IN.Lz / IN.lb))
    #
    #
    #
    # starting from (fcc) lattice
    IN.fcc()
    pas = 0
    #write_input(IN, N, pas, conf_out=dir+"conf_fcc0984.b")
    writexyz(IN, N, Na)
    #
    #
    #
    #
    mode=2
    #pas = read_input(IN, N, conf_in=dir+"conf_fcc0984.b")
    t0 = process_time()    
    (enep, enek, virial, vcmx, vcmy, vcmz) = IN.eqmdi( N, Na, mx, my, mz, kt, pas, mode)
    print( "# initial conditions step = %d, cpu = %8.4f" % (pas, process_time()-t0) )
    #
    #
    write_input(IN, N, nstep, conf_out=dir+"conf_fcc0984.b")
    #
    #
    #
    #
    nstep= 10      #  number of multiples of 'freq' time steps 
    # Equilibrium run with temperature from maxwellian sampling
    print("# starting eqmd trajectory\n")
    print( "    'pas'   'enep'     'enek'   'enet'      'virial'   'vcm' ")
    print(" %9.2f %9.4f %9.4f %12.7f %8.3f %12.2g %7.2g %7.2g" % (pas*dt,enep, enek, enep+enek, virial, vcmx, vcmy, vcmz) )
    t0 = process_time()
    for it in range(nstep):
        (t, enep, enek, virial, vcmx, vcmy, vcmz) = IN.eqmdr( N, Na, mx, my, mz, kt, pas, dt, freq)
        print(" %9.2f %9.4f %9.4f %12.7f %8.3f %12.2g %7.2g %7.2g" % (t,enep, enek, enep+enek, virial, vcmx, vcmy, vcmz) )
        pas += freq    
    print( "# (IN)-Equilibrium run, cpu = %8.4f" % (process_time()-t0) )
    g=3*N-3
    nstep *= freq
    print( "# ending initial equilibrium run  <ep>=%6.1f  <ek>=%7.4f   T=%8.3f  P=%8.3f\n" % ( IN.ept/nstep, IN.ekt/nstep, 2.*IN.ekt/(nstep*g), IN.pres/nstep) )
    # final positions and momenta to file conf_in.b
    write_input(IN, N, nstep, conf_out=dir+"conf_eg0984.b")
    writexyz(IN, N, Na)
