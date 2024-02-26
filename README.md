The goal of this weekend project is to get acquainted with building distributed systems from scratch and using pytorch+threading to massively speed up a rendering task.

The project has two components:

## Explorer
A simple UI that renders a piece of the mandelbrot set in low-resolution, allowing me to iteratively select views to "dive" into.

Dives bottom out around 1e-15 zoom.

The UI displays the coordinates of zones of interest and saves all generated images for later perusal.

## Animator
A program to render a zoom across multiple GPUs. The program is pixel-parallel using pytorch and frame-parallel (across GPUs) using threading.

The animator implements load balancing and graceful resumption, and can recognize when frames are missing and "heal" the video.

Frame-stitching was done as an after-processing step with ffmpeg.


## Performance
On an 8xa100_40gb cluster, we can render (on average) 1 frame/8seconds @ 8k resolution.

This means a 10 second zoom @60fps will take a bit more than an hour to render.

In terms of zoom performance, things start to deteriorate at 1e-15 diameter. I think this is due to the 64 bit floating point numbers losing expressivity around here.

I wonder if there's a way to remap the origin on the complex plane in order to help us use the more expressive part of the 64 bit floating point range. (apparently no)

The state of the art here is the "perturbation calculation algorithm" [(link)](https://www.ultrafractal.com/help/index.html?/help/formulas/perturbationcalculations.html) which calculates reference orbits using high precision (> 64 bit) and then calculates the rest as perturbations from the reference orbit. 

I feel like this project is wrapping up so I likely won't implement this but it's good to know how it works.

## Gallery

<img width="1153" alt="image" src="https://github.com/milocress/mandeltorch/assets/19612401/79a9943a-857e-4aa6-b804-7daeb266d1d4">
<img width="1186" alt="image" src="https://github.com/milocress/mandeltorch/assets/19612401/05dba356-cd83-4702-92d1-23b79e043e83">
<img width="1547" alt="image" src="https://github.com/milocress/mandeltorch/assets/19612401/53bd283a-0889-4727-b2fe-9ae3289eb92e">
<img width="762" alt="image" src="https://github.com/milocress/mandeltorch/assets/19612401/078d7377-056c-4e77-bf5d-ec21e98f1cd8">
<img width="1211" alt="image" src="https://github.com/milocress/mandeltorch/assets/19612401/756da9b3-e176-41f5-9966-bcd30f70491c">
<img width="1295" alt="image" src="https://github.com/milocress/mandeltorch/assets/19612401/996dbfcf-a5c4-4c48-8704-448225a34fb8">




