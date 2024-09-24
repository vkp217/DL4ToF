import numpy as np
# from helpers import convolve_3d_fft, convolve_3d_elementwise, add_poisson_noise
from fli_generator import FliDataGenerator
import matplotlib.pyplot as plt

def generate_dummy_data():
    img_height, img_width = 28, 28  # Example dimensions for the 2D image
    irf_depth = 70  # Example depth for the 3D IRF array

    # Create random IRF with shape (depth, height, width)
    irf = np.random.rand(170, 250, irf_depth)  # 5x5 IRF with a depth of 50

    # Create random image with shape (height, width)
    img = np.random.rand(img_height, img_width)

    # Define tau1, tau2, and frac1 as lists with two elements
    tau1 = [0.1, 0.6]  # Example range for tau1
    tau2 = [0.6, 2]  # Example range for tau2
    frac1 = [0.0, 0.6]  # Example range for frac1

    # Define gate width and gate
    gate_width = 0.04  # Example gate width in nanoseconds
    gate = irf_depth  # Set gate to match IRF depth

    return irf, img, tau1, tau2, frac1, gate_width, gate

def main():
    irf, img, tau1, tau2, frac1, gate_width, gate = generate_dummy_data()
    # Initialize the FliDataGenerator class
    fli_generator = FliDataGenerator(irf, img, tau1, tau2, frac1, gate_width, gate, noise_=True)

    # Generate decay parameters
    stack_parameters = fli_generator.decay_gen()

    # Print the shapes of the generated outputs
    print("Stack Parameters Shape:", stack_parameters.shape)
    print("IRF Out Shape:", fli_generator.irf_out.shape)
    print("A Shape:", fli_generator.A.shape)
    print("B Shape:", fli_generator.B.shape)
    print("Decay Shape:", fli_generator.dec.shape)
    print("Convolution Shape:", fli_generator.dec_conv.shape)

    plt.plot(np.squeeze(fli_generator.dec_conv[np.random.randint(28),np.random.randint(28),:]))
    plt.show()

if __name__ == "__main__":
    main()
