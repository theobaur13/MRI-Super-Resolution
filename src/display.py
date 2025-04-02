import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_matrix(ax, matrix, slice=65, cmap='gray', axis="z"):
    # set NaN values to red on colormap
    cmap = plt.cm.gray
    cmap.set_bad(color='red')
    
    matrix = matrix.real
    if axis == 0:
        matrix = matrix[slice, :, :]
    elif axis == 1:
        matrix = matrix[:, slice, :]
    elif axis == 2:
        matrix = matrix[:, :, slice]
        
    ax.imshow(matrix, cmap=cmap)
    # ax.axis('off')

def display_simulated_comparison(image, simulated_image, kspace, simulated_kspace, axis=0, highlight=False, show_kspace=True, show_image=True):
    if show_image:
        fig_images, axes_images = plt.subplots(1, 2, figsize=(9, 4))
        
        plot_matrix(axes_images[0], image, axis=axis)
        axes_images[0].set_title('Original Image (3T)')
        
        plot_matrix(axes_images[1], simulated_image, axis=axis)
        axes_images[1].set_title('Simulated Image (1.5T)')

        plt.tight_layout()
        plt.show(block=False)  
    
    if show_kspace:
        fig_kspace, axes_kspace = plt.subplots(1, 2, figsize=(9, 4))

        plot_matrix(axes_kspace[0], kspace, axis=axis)
        axes_kspace[0].set_title('Original k-space')
        
        if highlight:
            simulated_kspace = jnp.where(simulated_kspace == 0, jnp.nan, simulated_kspace)
        plot_matrix(axes_kspace[1], simulated_kspace, axis=axis)
        axes_kspace[1].set_title('Simulated k-space')
        
        plt.tight_layout()
        plt.show()

def display_t1_5_vs_t3(t1_5, t3, axis=0):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    
    plot_matrix(axes[0], t3, axis=axis)
    axes[0].set_title('3T Image')

    plot_matrix(axes[1], t1_5, axis=axis)
    axes[1].set_title('1.5T Image')

    
    plt.tight_layout()
    plt.show(block=False)