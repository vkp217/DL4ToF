import numpy as np
import sys
import os

# sys.path.append(os.path.join(os.getcwd(),'src'))


from .helpers import  norm3D, convolve_3d_fft, convolve_3d_elementwise, add_poisson_noise


class FliDataGenerator:
    def __init__(self, irf, img, tau1, tau2, frac1, gate_width, gate, noise_=True):
        # Validate irf as a 3D array
        self.irf = self._validate_array(irf, expected_dim=3, name="irf")
        
        # Validate img as a 2D array
        self.img = self._validate_array(img, expected_dim=2, name="img")

        # Validate tau1, tau2, frac1 lists
        self.tau1 = self._validate_list(tau1, "tau1")
        self.tau2 = self._validate_list(tau2, "tau2")
        self.frac1 = self._validate_list(frac1, "frac1")
        
        # Assign gate and gate_width
        self.gate_width = gate_width
        self.gate = gate
        self.irf_out = None # Initialize as None for later use
        self.noise = noise_

    def decay_gen(self):  # IRF, Gate_width (in ns)
        a, b = self.img.shape
        c = self.irf.shape[2]

        # Create random index arrays for selecting from `IRF pixels`
        rand_i = np.random.randint(0, self.irf.shape[0], size=(a, b))  # Random indices for dimension 0
        rand_j = np.random.randint(0, self.irf.shape[1], size=(a, b))  # Random indices for dimension 1

        # Create random tau1, tau2, frac1 within the range provided in (a, b) shape
        tau_1 = np.random.uniform(low=self.tau1[0], high=self.tau1[1], size=(a, b))
        tau_2 = np.random.uniform(low=self.tau2[0], high=self.tau2[1], size=(a, b))
        frac_1 = np.random.uniform(low=self.frac1[0], high=self.frac1[1], size=(a, b))

        # Generating normalized irf_out
        self.irf_out = norm3D(self.irf[rand_i, rand_j], axis=2)  # Normalized IRF

        # Time points
        if c == self.gate:
            t = np.arange(0, c) * self.gate_width * 1e-3
            t_minus = -t
            t_3d = np.tile(t_minus, (a, b, 1))
        else:
            raise ValueError(f"Gate width {self.gate} doesn't match with IRF gate dimension.")

        # Initialize arrays
        self.A = np.exp(t_3d / tau_1[:, :, np.newaxis]) * frac_1[:, :, np.newaxis]
        self.B = np.exp(t_3d / tau_2[:, :, np.newaxis]) * (1 - frac_1[:, :, np.newaxis])
        self.dec = norm3D(self.A + self.B, axis=2)  # Normalizing the summation of exponential decays

        # Convolutions
        dec_conv1 = convolve_3d_fft(self.irf_out, self.dec, axis=2)


        # dec_conv2 = convolve_3d_elementwise(self.irf_out, self.dec)

        # if not np.array_equal(dec_conv1, dec_conv2):
        #     raise ValueError("Convolution aborted! Both methods are generating different values.")


        
        self.dec_conv = dec_conv1

        if self.noise:
            self.dec_conv = add_poisson_noise(self.dec_conv, lam=2, axis=2)

        # Stack results
        self.stack_decays = np.stack([self.A, self.B, self.dec, self.irf_out, self.dec_conv], axis=3)
        stack_parameters = np.stack([tau_1, tau_2, frac_1, 1 - frac_1], axis=2)
        return stack_parameters
        
    def _validate_list(self, param, name):
        """Validate that param is a list with exactly 2 elements."""
        if isinstance(param, list) and len(param) == 2:
            return param
        raise ValueError(f"Expected {name} to be a list with exactly 2 elements.")

    def _validate_array(self, param, expected_dim, name):
        """Validate that param is a numpy array with the expected number of dimensions."""
        if isinstance(param, np.ndarray) and param.ndim == expected_dim:
            return param
        raise ValueError(f"Expected {name} to be a {expected_dim}D numpy array.")