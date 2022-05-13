# Ising simulator modified from https://github.com/bdhammel/ising-model

import numpy as np



class IsingLattice:

    def __init__(self, initial_state, size, J, h0, run_id, mask):
        self.size = size
        
        if h0 is np.ndarray:
            self.h = h0
        else:
            self.h = h0*np.ones((size,size))
            
        if J is np.ndarray:
            self.J = J
        else:
            self.J = J*np.ones((size,size))
            
        self.run_id = run_id
        if mask is None:
            self.mask = np.ones((size,size))
        else:
            self.mask = mask
            
        self.system = self._build_system(initial_state)

    @property
    def sqr_size(self):
        return (self.size, self.size)

    def _build_system(self, initial_state):
        """Build the system

        Build either a randomly distributed system or a homogeneous system (for
        watching the deterioration of magnetization

        Parameters
        ----------
        initial_state : str: "r" or other
            Initial state of the lattice.  currently only random ("r") initial
            state, or uniformly magnetized, is supported
        """

        if initial_state == 'r':
            try:
                rng = np.random.default_rng()
            except AttributeError:
                rng = np.random.RandomState()
            system = rng.choice([-1, 1], self.sqr_size)
        elif initial_state == 'u':
            system = np.ones(self.sqr_size)
        elif initial_state == 'r_h':
            est_mag = np.tanh(2*np.mean(self.h)) #rough estimate of magnetization from mean field approx
            prob = (est_mag + 1)/2
            system = np.random.choice([-1, 1], size=self.sqr_size, p=[1-prob,prob])
        else:
            raise ValueError(
                "Initial State must be 'r', random, or 'u', uniform, or 'r_h', random with estimated magnetization"
            )
        system = np.multiply(system,self.mask) # zero out masked values
        
        return system

    def _bc(self, i):
        """Apply periodic boundary condition

        Check if a lattice site coordinate falls out of bounds. If it does,
        apply periodic boundary condition

        Assumes lattice is square

        Parameters
        ----------
        i : int
            lattice site coordinate

        Return
        ------
        int
            corrected lattice site coordinate
        """
        if i >= self.size:
            return 0
        if i < 0:
            return self.size - 1
        else:
            return i

    def energy(self, N, M):
        """Calculate the energy of spin interaction at a given lattice site
        i.e. the interaction of a Spin at lattice site n,m with its 4 neighbors

        - S_n,m*(S_n+1,m + Sn-1,m + S_n,m-1, + S_n,m+1)

        Parameters
        ----------
        N : int
            lattice site coordinate
        M : int
            lattice site coordinate

        Return
        ------
        float
            energy of the site
        """
        return -2*self.J[N,M]*self.system[N, M]*(
            self.system[self._bc(N - 1), M] + self.system[self._bc(N + 1), M]
            + self.system[N, self._bc(M - 1)] + self.system[N, self._bc(M + 1)]
        ) - np.multiply(self.h,self.system[N, M])

    @property
    def internal_energy(self):
        # e = 0
        # E = 0
        # E_2 = 0
        
        a = self.system
        
        
        E_mat = -np.multiply(self.J,np.multiply(a,(np.roll(a,-1,0)+np.roll(a,1,0)+np.roll(a,-1,1)+np.roll(a,1,1)))) - np.multiply(self.h,a)
        
        E = np.sum(E_mat)
        E_2 = np.sum(np.power(E_mat,2))
        
        # for i in range(self.size):
        #     for j in range(self.size):
        #         e = self.energy(i, j)
        #         E += e
        #         E_2 += e**2

        U = (1./self.size**2)*E
        U_2 = (1./self.size**2)*E_2
        
        return U, U_2

    def heat_capacity(self, temp):
        U, U_2 = self.internal_energy
        return np.mean((U_2 - U**2)/np.power(temp,2)) # if temp is matrix, this returns mean heat capacity over the lattice

    @property
    def magnetization(self):
        """Find the overall magnetization of the system
        """
        return np.sum(self.system)/self.size**2


def run(lattice, temps, fields, burn_time, epoch_len, bias):
    """Run the simulation
    """
    
    
    epochs = temps.shape[0]
    try:
        rng = np.random.default_rng()
    except AttributeError:
        rng = np.random.RandomState()
    
    sys_array = np.zeros((epochs,lattice.system.shape[0],lattice.system.shape[1]))
    magnetization = np.zeros(epochs)
    heat_capacity = np.zeros(epochs)
    
    sys_array_burn = np.zeros((burn_time,lattice.system.shape[0],lattice.system.shape[1]))
    magnetization_burn = np.zeros(burn_time)
    heat_capacity_burn = np.zeros(burn_time)
    
    coord_list = []
    for N in range(lattice.size):
        for M in range(lattice.size):
            if lattice.system[N,M] != 0:
                coord_list.append((N,M))

    for step in range(epochs+burn_time):
        
        if step < burn_time:
            epoch = 0
            if temps.ndim == 1:
                this_temp_interval = temps[0]*np.ones(2)
            else:
                this_temp_interval = [temps[0,:,:],temps[0,:,:]]
            if fields.ndim == 1:
                this_field_interval = fields[0]*np.ones(2)
            else:
                this_field_interval = [fields[0,:,:],fields[0,:,:]]
            if step % 10 == 0:
                print('Burn epoch ' + str(step) + '/' + str(burn_time))
        else:
            epoch = step - burn_time
            if temps.ndim == 1:
                dtemp = temps[1]-temps[0]
                this_temp_interval = temps[epoch] + np.array([0,dtemp])
            else:
                this_temp_interval = [temps[epoch,:,:],temps[epoch,:,:]] # don't bother with subdivisions here
            
            if fields.ndim == 1:
                dh = fields[1]-fields[0]
                this_field_interval = fields[epoch] + np.array([0,dh])
            else:
                this_field_interval = [fields[epoch,:,:],fields[epoch,:,:]]
            if epoch % 10 == 0:
                print('Run epoch ' + str(epoch) + '/' + str(epochs))
            
        if temps.ndim == 1:
            subtemps = np.linspace(this_temp_interval[0],this_temp_interval[1],epoch_len)
            
        if fields.ndim == 1:
            subfields = np.linspace(this_field_interval[0],this_field_interval[1],epoch_len)
        
        step_avg = np.zeros((lattice.size,lattice.size))
        for substep in range(epoch_len):
            # Randomly select an unmasked site on the lattice
            N, M = coord_list[np.random.randint(len(coord_list))]
            
            
            if temps.ndim == 1:
                this_temp = subtemps[substep]
            else:
                this_temp = temps[epoch,:,:]
            
            if fields.ndim == 1:
                lattice.h = subfields[substep]
            else:
                lattice.h = fields[epoch,:,:]

            # Calculate energy of a flipped spin
            E = -1*lattice.energy(N, M)
            
            
            # "Roll the dice" to see if the spin is flipped
            if E <= 0.:
                lattice.system[N, M] *= -1
            elif np.exp(-E/this_temp[N,M]) > rng.uniform() + bias:
                lattice.system[N, M] *= -1
            
            # E = lattice.energy(N,M)
            # lattice.system[N,M] *= -1 #trial spin flip
            # Ef = lattice.energy(N,M)
            # if E - Ef < 0:
            #     PA = 1
            # else:
            #     PA = np.exp((E-Ef)/this_temp)
            #     if PA < rng.uniform():
            #         lattice.system[N,M]*=-1 #flip back
            
            step_avg += lattice.system
        
        step_avg = step_avg/epoch_len
        
        
        if step >= burn_time:
            sys_array[epoch,:,:] = step_avg
            magnetization[epoch] = lattice.magnetization
            heat_capacity[epoch] = lattice.heat_capacity(this_temp)
        else:
            sys_array_burn[step,:,:] = step_avg
            magnetization_burn[step] = lattice.magnetization
            heat_capacity_burn[step] = lattice.heat_capacity(this_temp)
    
    out_vars = {'sys':sys_array, 'magnetization':magnetization, 'heat_capacity':heat_capacity,
                'sys_burn':sys_array_burn, 'magnetization_burn':magnetization_burn, 'heat_capacity_burn':heat_capacity_burn}
    return out_vars



def ising_run(temps, fields, size, J, run_id, burn_time, epoch_len, bias, initial_state='r', mask=None):
    
    if fields.ndim == 1:
        h0 = fields[0]
    else:
        h0 = fields[0,:,:]
    
    lattice = IsingLattice(initial_state=initial_state, size=size, J=J, run_id=run_id, h0 = h0, mask=mask)
    out_vars = run(lattice, temps, fields, burn_time, epoch_len, bias)

    return out_vars

