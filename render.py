from typing import Optional
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from math import pi
import os
from datetime import datetime
from PIL import Image

save_root = '/mnt/workdisk/iamroot/work/mandelbrot/'


def mandel(
        max_iter: int = 200, 
        scale_up: int = 1, 
        mid_left: tuple = (-2.5, 0.0), 
        x_range: float = 4, 
        dtype: torch.dtype = torch.float64, 
        device='cpu', 
        spectrum_compression_factor: int = 1600,
        frame_id: Optional[int] = None,
        video_name: Optional[str] = None
        ):
    if video_name is None:
        video_name = 'default'
    # Set the size of the image
    width, height = int(3840 * scale_up), int(2160 * scale_up)  # 4K resolution

    x_left, y_middle = mid_left

    aspect_ratio = width / height
    y_range = x_range / aspect_ratio  # Scale y-range based on aspect ratio
    x_min, x_max = x_left, x_left + x_range
    y_min, y_max = y_middle - y_range / 2, y_middle + y_range / 2

    # Create a meshgrid of complex numbers
    x = torch.linspace(x_min, x_max, width, dtype=dtype, device=device)
    y = torch.linspace(y_min, y_max, height, dtype=dtype, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    Z = X + 1j * Y

    # Initialize the image
    image = torch.zeros(height, width, dtype=dtype, device=device)

    # Compute the Mandelbrot set
    C = Z.clone()
    for i in range(max_iter):
        Z = Z**2 + C
        mask = (torch.abs(Z) >= 2) & (image == 0)
        image[mask] = i

    # Color rendering stuff
    image = image * 2 * pi / 1600
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
        directory = os.path.join(save_root, 'captures')
        video_dir = os.path.join(save_root, 'videos', video_name)
        os.makedirs(directory, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

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

        if frame_id is not None:
            image_path = os.path.join(video_dir, f'frame_{frame_id}.png')
            pal = cmap2palette('inferno')
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

frames_per_second = 60
seconds = 20

outer_spectrum_compression_factor = 1200
inner_spectrum_compression_factor = 1200 # keep it constant
zoom_outer_diameter = 1
# (-0.28440534236668047, -0.6426373874281622) 1.2463488835009854e-12
zoom_target_diameter = 1.2463488835009854e-12
zoom_target_center = (-0.28440534236668047, -0.6426373874281622)

zoom_target_center = (zoom_target_center[0] - zoom_target_diameter / 2, zoom_target_center[1])

def log_interpolate(big_num, small_num, progress):
    log_big_num = np.log(big_num)
    log_small_num = np.log(small_num)

    log_interpolated = log_big_num - progress * (log_big_num - log_small_num)
    return np.exp(log_interpolated)

total_frames = frames_per_second * seconds

video_name = 'back_to_black'

os.makedirs(os.path.join(save_root, 'videos', video_name), exist_ok=True)
existing_frames = os.listdir(os.path.join(save_root, 'videos', video_name))
existing_frame_indices = set([int(frame.split('_')[1].split('.')[0]) for frame in existing_frames if frame.startswith('frame')])

frames_per_device = 1 # trial and error, adjust this when OOM
num_ranks = torch.cuda.device_count() * frames_per_device

def render_video_frames(rank):
    for frame in tqdm(range(total_frames)):
        if frame % num_ranks != rank:
            continue

        elif frame in existing_frame_indices:
            print(f"Frame {frame} already exists")
            continue

        print(f"Frame {frame}")
        interpolated_diameter = log_interpolate(zoom_outer_diameter, zoom_target_diameter, frame / total_frames)
        mid_left = (zoom_target_center[0] - interpolated_diameter / 2, zoom_target_center[1])
        interpolated_spectrum_compression_factor = outer_spectrum_compression_factor + (inner_spectrum_compression_factor - outer_spectrum_compression_factor) * frame / total_frames
        device = rank % torch.cuda.device_count()
        args = {
            'max_iter': 12800,
            'scale_up': 2,
            'mid_left': mid_left,
            'x_range': interpolated_diameter,
            'dtype': torch.float64,
            'device': f'cuda:{device}',
            'spectrum_compression_factor': interpolated_spectrum_compression_factor,
            'frame_id': frame,
            'video_name': video_name
        }
        print(args)
        mandel(**args)

        print("Frame rendered")

with torch.multiprocessing.Pool(num_ranks) as pool:
    pool.map(render_video_frames, range(num_ranks))