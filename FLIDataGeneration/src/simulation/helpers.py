import numpy as np
from scipy.fft import fft, ifft 
from scipy.signal import convolve

def norm3D(A, axis=2):
    A_sum = np.sum(A, axis=axis, keepdims=True)  
    A_sum[A_sum == 0] = 1
    A_normalized = A / A_sum
    return A_normalized

def convolve_3d_fft(A, B, axis = 2):
    """
    Parameters:
    A (numpy.ndarray): First 3D matrix of shape (M, N, P).
    B (numpy.ndarray): Second 3D matrix of shape (M, N, P).

    Returns:
    numpy.ndarray: Convolved 3D matrix of the same shape as A and B.
    """
    
    # Perform FFT along axis 2 for both A and B
    A_fft = fft(A, axis=axis)
    B_fft = fft(B, axis=axis)

    # Element-wise multiplication in the frequency domain
    C_fft = A_fft * B_fft

    # Perform inverse FFT to get the convolved result
    C = np.real(ifft(C_fft, axis=axis))

    # Keep the same shape as A and B
    result = C[:, :, :A.shape[axis]]  # Ensures the output has the same size

    return result

def convolve_3d_elementwise(A, B):
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    
    # Initialize the result matrix with the same shape as A
    result = np.zeros_like(A)

    # Perform convolution along axis 2 for each (i, j) slice
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            result[i, j, :] = convolve(A[i, j, :], B[i, j, :], mode='same')

    return result

def add_poisson_noise(data, lam=1.0, axis=2):
    """
    Add Poisson noise to a 3D array along a specified axis.

    Parameters:
    - data: 3D numpy array
    - lam: Rate (lambda) parameter for the Poisson distribution
    - axis: Axis along which to add noise (default is 2)

    Returns:
    - Noisy 3D numpy array
    """
    # Generate Poisson noise with the same size as the specified axis
    noise_shape = [1] * data.ndim  # Start with a shape of ones for broadcasting
    noise_shape[axis] = data.shape[axis]  # Set the size for the specified axis

    # Generate noise and reshape it for broadcasting
    noise = np.random.poisson(lam, noise_shape)

    # Add noise to the data
    noisy_data = data + noise

    return noisy_data