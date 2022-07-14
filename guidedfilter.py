import numpy as np

def box(img, radius):

    (rows, cols) = img.shape[:2]
    dst = np.zeros_like(img)

    seg = [1] * img.ndim
    seg[0] = radius
    sum = np.cumsum(img, 0)
    dst[0:radius+1, :, ...] = sum[radius:2*radius+1, :, ...]
    dst[radius+1:rows-radius, :, ...] = sum[2*radius+1:rows, :, ...] - sum[0:rows-2*radius-1, :, ...]
    dst[rows-radius:rows, :, ...] = np.tile(sum[rows-1:rows, :, ...], seg) - sum[rows-2*radius-1:rows-radius-1, :, ...]

    seg = [1] * img.ndim
    seg[1] = radius
    sum = np.cumsum(dst, 1)
    dst[:, 0:radius+1, ...] = sum[:, radius:2*radius+1, ...]
    dst[:, radius+1:cols-radius, ...] = sum[:, 2*radius+1 : cols, ...] - sum[:, 0 : cols-2*radius-1, ...]
    dst[:, cols-radius: cols, ...] = np.tile(sum[:, cols-1:cols, ...], seg) - sum[:, cols-2*radius-1 : cols-radius-1, ...]

    return dst


def guided_filter(input, filter, radius, eps):

    (rows, cols) = input.shape

    CNT = box(np.ones([rows, cols]), radius)

    mean_input = box(input, radius) / CNT
    mean_filter = box(filter, radius) / CNT

    a = ((box(input * filter, radius) / CNT) - mean_input * mean_filter) / (((box(input * input, radius) / CNT) - mean_input * mean_input) + eps)
    b = mean_filter - a * mean_input

    return (box(a, radius) / CNT) * input + (box(b, radius) / CNT)
