import numpy as np
import yaff
import molmod


def transform_lower_triangular(pos, rvecs, reorder=False):
    """Transforms coordinate axes such that cell matrix is lower diagonal
    The transformation is derived from the QR decomposition and performed
    in-place. Because the lower triangular form puts restrictions on the size
    of off-diagonal elements, lattice vectors are by default reordered from
    largest to smallest; this feature can be disabled using the reorder
    keyword.
    The box vector lengths and angles remain exactly the same.
    Parameters
    ----------
    pos : array_like
        (natoms, 3) array containing atomic positions
    rvecs : array_like
        (3, 3) array with box vectors as rows
    reorder : bool
        whether box vectors are reordered from largest to smallest
    """
    if reorder: # reorder box vectors as k, l, m with |k| >= |l| >= |m|
        norms = np.linalg.norm(rvecs, axis=1)
        ordering = np.argsort(norms)[::-1] # largest first
        a = rvecs[ordering[0], :].copy()
        b = rvecs[ordering[1], :].copy()
        c = rvecs[ordering[2], :].copy()
        rvecs[0, :] = a[:]
        rvecs[1, :] = b[:]
        rvecs[2, :] = c[:]
    q, r = np.linalg.qr(rvecs.T)
    flip_vectors = np.eye(3) * np.diag(np.sign(r)) # reflections after rotation
    rotation = np.linalg.inv(q.T) @ flip_vectors # full (improper) rotation
    pos[:]   = pos @ rotation
    rvecs[:] = rvecs @ rotation
    assert np.allclose(rvecs, np.linalg.cholesky(rvecs @ rvecs.T), atol=1e-6)
    rvecs[0, 1] = 0
    rvecs[0, 2] = 0
    rvecs[1, 2] = 0
    

class RectangularMonteCarloBarostat(yaff.VerletHook):
    """Monte Carlo algorithm for pressure control"""

    def __init__(self, temperature, pressure, start=0, step=10):
        """Constructor

        Arguments
        ---------

        temperature (float):
            temperature at which the simulation is performed, in kelvin.

        pressure (float):
            external (and isotropic) pressure that is applied on the system,
            in atomic units.

        start (int, default 0):
            starting point during the Verlet run from which MC trial moves in
            the cell degrees of freedom should be performed.

        step (int, default 10):
            period between consecutive trial moves.
        """
        self.beta     = 1 / (molmod.constants.boltzmann * temperature)
        self.pressure = pressure

        #self.internal_cell    = np.zeros((3, 3)) # internal representation of cell
        self.internal_ampl    = 0 # max trial amplitudes for isotropic and anisotropic moves
        self.internal_trials  = 0 # counts trials for isotropic and anisotropic moves
        self.internal_accepts = 0 # counts accepted trials for isotropic and anisotropic moves

        self.econs_correction = 0
        self.start = start
        self.step  = step

    def init(self, iterative):
        """Initialize cell and radii"""
        rvecs = iterative.ff.system.cell._get_rvecs().copy()
        # compute internal cell representation and initialize amplitudes
        # amplitude for isotropic moves is set to 10% of the current volume
        # amplitude for anisotropic moves is set to 0.01
        self.internal_ampl = 0.1 * (np.linalg.det(rvecs)) ** (1/3) # initialize nonzero for all 9 amplitudes

    def pre(self, iterative, chainvel0=None):
        pass

    def post(self, iterative, chainvel0=None):
        """Generates and applies trial moves

        The kind of trial move is chosen randomly depending on the mode of the
        barostat:

            full:
                includes both isotropic and anisotropic trial moves

            anisotropic:
                includes only anisotropic trial moves

            isotropic:
                includes only isotropic trial moves

        """
        # update internal_cell with current rvecs
        trial = self.get_trial(iterative.ff.system.cell._get_rvecs()) # generates a trial move
        accepted = self.perform_trial(iterative, trial) # makes an attempt
        self.internal_trials += 1
        if accepted:
            self.internal_accepts += 1
        # maintain desired acceptance ratio around 50% by adjusting the
        # amplitudes based on trial and acceptance statistics
        self.adjust_amplitudes()

    def get_trial(self, rvecs):
        """Generates a trial move"""
        trial = rvecs.copy()
        # generate triangular trial
        delta = 2 * self.internal_ampl * (np.random.uniform(size=(3, 3)) - 0.5)
        delta[0, 1] = 0
        delta[0, 2] = 0
        delta[1, 2] = 0
        trial += delta
        for i in range(3): # flip box vectors if necessary
            if trial[i, i] < 0:
                trial[i, :] *= (-1.0)
                print('flipping box vector {}'.format(i))
        # to reduced form
        trial[2, :] = trial[2, :] - trial[1, :] * np.round(trial[2, 1] / trial[1, 1])
        trial[2, :] = trial[2, :] - trial[0, :] * np.round(trial[2, 0] / trial[0, 0])
        trial[1, :] = trial[1, :] - trial[0, :] * np.round(trial[1, 0] / trial[0, 0])
        return trial

    def perform_trial(self, iterative, trial):
        """Performs a trial move

        Arguments
        ---------

        iterative (``Iterative`` instance):
            iterative containing the ``ForceField`` instance

        frac (2darray):
            contains fractional coordinates of all atoms

        trial (1darray of length 6):
            contains trial cell in the internal representation

        """
        rvecs  = iterative.ff.system.cell._get_rvecs().copy()
        assert rvecs[0, 1] == 0
        assert rvecs[0, 2] == 0
        assert rvecs[1, 2] == 0
        pos    = iterative.ff.system.pos.copy()
        # translate particles to first rectangular(!) periodic box
        for i in range(pos.shape[0]):
            pos[i, :] -= rvecs[2, :] * np.floor(pos[i, 2] / rvecs[2, 2])
            pos[i, :] -= rvecs[1, :] * np.floor(pos[i, 1] / rvecs[1, 1])
            pos[i, :] -= rvecs[0, :] * np.floor(pos[i, 0] / rvecs[0, 0])
        for i in range(pos.shape[0]):
            assert np.all(np.abs(pos[i, :]) < np.diag(rvecs))

        # scale rectangular axes
        scale_x = trial[0, 0] / rvecs[0, 0]
        scale_y = trial[1, 1] / rvecs[1, 1]
        scale_z = trial[2, 2] / rvecs[2, 2]
        pos_trial = pos.copy()
        pos_trial[:, 0] *= scale_x
        pos_trial[:, 1] *= scale_y
        pos_trial[:, 2] *= scale_z

        iterative.ff.update_rvecs(trial) # update force field with trial
        iterative.ff.update_pos(pos_trial)

        E0 = iterative.epot # initial potential energy
        E1 = iterative.ff.compute() # trial potential energy
        V0 = np.linalg.det(rvecs)
        V1 = np.linalg.det(trial) # new volume
        J1 = trial[0, 0] ** 2 * trial[1, 1] # a_x ** 2 * b_y
        J0 = rvecs[0, 0] ** 2 * rvecs[1, 1]
        natom = pos.shape[0]
        ndim  = pos.shape[1] # number of dimensions is always 3

        # the trial is accepted based on an acceptance ratio of the form
        # np.exp(- exponent), where exponent is an expression that contains
        # multiple contributions
        exponent = 0
        exponent -= self.beta * self.pressure * (V1 - V0) # volume change
        exponent -= self.beta * (E1 - E0) # energy change
        exponent += (natom - 2) * np.log(V1 / V0) # jacobian
        exponent += np.log(J1 / J0) # jacobian
        #print('volume: ', np.linalg.det(trial) / molmod.units.angstrom ** 3)
        #print('PV contrib: ', self.beta * self.pressure * (V1 - V0))
        #print('jacobian contrib: ', (natom - 2) * np.log(V1 / V0) * np.log(J1 / J0))
        #print('{}   /   {}'.format(self.internal_accepts, self.internal_trials))
        #accepted = (exponent < 0) or (np.random.uniform(0, 1) < np.exp(exponent))
        accepted = np.random.uniform(0, 1) < np.exp(exponent)

        if accepted: # update iterative gpos and acceleration if accepted
            iterative.pos[:]  = pos_trial
            iterative.rvecs[:] = trial
            iterative.gpos[:] = 0.0
            iterative.ff.update_pos(pos_trial)
            iterative.ff.update_rvecs(trial)
            iterative.epot = iterative.ff.compute(iterative.gpos)
            iterative.acc  = - iterative.gpos / iterative.masses.reshape(-1,1)
        else: # revert iterative if not accepted
            iterative.ff.update_pos(pos)
            iterative.ff.update_rvecs(rvecs)
        return accepted

    def adjust_amplitudes(self):
        """Adjust amplitudes based on the trial/acceptance statistics"""
        if self.internal_trials >= 10:
            if self.internal_accepts < 0.25 * self.internal_trials:
                self.internal_ampl /= 1.1
                self.internal_accepts = 0
                self.internal_trials = 0
            elif self.internal_accepts > 0.75 * self.internal_trials:
                self.internal_ampl *= 1.1
                self.internal_accepts = 0
                self.internal_trials = 0
