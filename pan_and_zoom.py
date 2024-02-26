from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from math import pi
import os
from datetime import datetime
from PIL import Image


def mandel(max_iter: int = 200, scale_up: int = 1, mid_left: tuple = (-2.5, 0.0), x_range: float = 4, dtype: torch.dtype = torch.float64):
    # Set the size of the image
    width, height = int(3840 * scale_up), int(2160 * scale_up)  # 4K resolution

    x_left, y_middle = mid_left

    aspect_ratio = width / height
    y_range = x_range / aspect_ratio  # Scale y-range based on aspect ratio
    x_min, x_max = x_left, x_left + x_range
    y_min, y_max = y_middle - y_range / 2, y_middle + y_range / 2

    # Create a meshgrid of complex numbers
    x = torch.linspace(x_min, x_max, width, dtype=dtype)
    y = torch.linspace(y_min, y_max, height, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    Z = X + 1j * Y

    # Initialize the image
    image = torch.zeros(height, width, dtype=dtype)

    # Compute the Mandelbrot set
    C = Z.clone()
    for i in tqdm(range(max_iter)):
        Z = Z**2 + C
        mask = (torch.abs(Z) >= 2) & (image == 0)
        image[mask] = i

    # Color rendering stuff
    image = image * 2 * pi / 1200
    image = (torch.sin(image)**2)

    # Plot the image
    fig, ax = plt.subplots()
    image_numpy = image.cpu().numpy()

    def cmap2palette(cmapName='inferno'):
        """Convert a Matplotlib colormap to a PIL Palette"""
        cmap = plt.get_cmap(cmapName)
        palette = [int(x*255) for entry in cmap.colors for x in entry]
        return palette

    def save_image(image_numpy):
        # Create the directory if it doesn't exist
        directory = os.path.expanduser('~/Documents/mandelbrot/captures')
        os.makedirs(directory, exist_ok=True)

        # Generate the timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        cmaps = ['inferno', 'plasma', 'magma', 'viridis']
        # Save the image as PNG
        for cmap in cmaps:
            pal = cmap2palette(cmap)
            image_path = os.path.join(directory, f'{timestamp}_{cmap}.png')
            image = Image.fromarray((image_numpy * 255).astype('uint8'))
            image.putpalette(pal)
            image.save(image_path)


    save_image(image_numpy)
    print("Image saved")
    ax.imshow(image_numpy, cmap='inferno')
    ax.axis('off')

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        new_mid_left = (x_range * x1 / width + x_min, y_range * ((y1 + y2) / 2 / height) + y_min)
        new_x_range = x_range * abs(x2 - x1) / width
        plt.close()
        print(new_mid_left, new_x_range)
        mandel(max_iter, scale_up, new_mid_left, new_x_range, dtype=dtype)

    # Create a rectangle selector and connect it to the onselect function
    rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)

    plt.show()

    return image_numpy

mandel(max_iter=6400, scale_up=.25, dtype=torch.float64)