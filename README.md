# ImageCleaner

Exploring ways to remove moving objects in images to leave just background

methods:

- Median: Simply take median of RGB values accross all frames
- Median with affine correction: match points across frame and inverse the transformation before applying median
