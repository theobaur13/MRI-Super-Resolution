import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn

def convert_to_kspace(image):
    kspace = fftshift(fftn(ifftshift(image), axes=(0, 1, 2)))
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, [0, 1, 2])))
    return kspace

def convert_to_image(kspace):
    image = fftshift(ifftn(ifftshift(kspace), axes=(0, 1, 2)))
    image *= np.sqrt(np.prod(np.take(image.shape, [0, 1, 2])))
    return image

def undersampling(kspace, axis, factor, method="random"):
    if method == "random":
        return random_undersampling(kspace, factor)
    elif method == "cartesian":
        return cartesian_undersampling(kspace, axis, factor)
    elif method == "radial":
        return radial_undersampling(kspace, axis, factor)
    elif method == "variable_density":
        return variable_density_undersampling(kspace, factor)

def random_undersampling(kspace, factor=1.2):
    mask = np.random.choice([0, 1], size=kspace.shape, p=[1 - 1 / factor, 1 / factor])
    return kspace * mask

def cartesian_undersampling(kspace, axis, factor=3):
    # Create a mask that keeps every x-th line along the specified axis
    mask = np.zeros(kspace.shape)
    slices = [slice(None)] * 3
    slices[axis] = slice(None, None, factor)  # Slice every x-th line
    
    mask[tuple(slices)] = 1
    return kspace * mask

def radial_undersampling(kspace, axis, radius=50):
    # Create a mask that keeps only the center of the k-space, where the axis is the axis of slicing
    mask = np.zeros(kspace.shape)
    center = np.array(kspace.shape) // 2
    Z, Y, X = np.indices(kspace.shape)

    if axis == 0:       # Cylinder along the z-axis
        mask = np.where(np.sqrt((X - center[2])**2 + (Y - center[1])**2) <= radius, 1, 0)
    elif axis == 1:     # Cylinder along the y-axis
        mask = np.where(np.sqrt((X - center[2])**2 + (Z - center[0])**2) <= radius, 1, 0)
    elif axis == 2:     # Cylinder along the x-axis
        mask = np.where(np.sqrt((Y - center[1])**2 + (Z - center[0])**2) <= radius, 1, 0)
    
    return kspace * mask

def variable_density_undersampling(kspace, factor=1.2):
    # Chance of sampling a line is inversely proportional to its distance from the center
    center = np.array(kspace.shape) // 2
    Z, Y, X = np.indices(kspace.shape)
    distances = np.sqrt((X - center[2])**2 + (Y - center[1])**2 + (Z - center[0])**2)

    # Normalize distances to [0, 1]
    distances = distances / np.max(distances)

    # Sigmoid function
    ks = 10
    distances = 1 / (1 + np.exp(-ks * (distances - 0.5)))

    # Flatten
    probabilities = 1 - distances
    probabilities = probabilities / factor
    probabilities = np.clip(probabilities, 0, 1)
    mask = (np.random.rand(*kspace.shape) < probabilities).astype(int)

    return kspace * mask