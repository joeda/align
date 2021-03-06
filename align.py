#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import os
import csv
from PIL import Image

CSV_ENDING = ".csv"

def table_to_img(table, inv_res, dtype=np.float64):
    table[:, 0:2] *= inv_res
    table[:, 0:2] -= np.amin(table, axis=0)[0:2]
    idxs = table[:, 0:2].astype(int).tolist()
    print(table[:, 0:2].astype(int))
    rows, cols = zip(*idxs)
    vals = table[:, 2].tolist()
    sz = (np.unique(table[:, 0]).size, np.unique(table[:,1]).size)
    print("Size: {}x{}".format(sz[0], sz[1]))
    res = np.empty(sz, dtype)
    res[:] = np.NaN
    res[rows, cols] = vals
    return res

def read_data(dir, offsets):
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    imgs = []
    for f in files:
        splits = os.path.splitext(f)[0].split("_")
        if len(splits) < 2:
            raise ValueError("File name " + f + "  needs to have the resolution behind the last _ ")
        resolution = float(splits[-1])
        offset = offsets[os.path.splitext(f)[0]]
        if os.path.splitext(f)[1] == CSV_ENDING:
            arr = np.genfromtxt(os.path.join(dir, f), delimiter=',')
            arr = table_to_img(arr, 1000 / resolution) # Annahme: Angaben in µm
            # vermutlich musst du das noch flippen, siehe https://numpy.org/doc/stable/reference/generated/numpy.flip.html
        else:
            img_path = os.path.join(dir, f)
            if Image.open(img_path).mode == "F":
                color = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            else:
                color = cv2.imread(img_path)
            arr = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        #perform rotation
        rows, cols = arr.shape
        # cols-1 and rows-1 are the coordinate limits.
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), offset[2], 1)
        arr = cv2.warpAffine(arr, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=np.NaN)
        imgs.append((arr, resolution, offset))
    return imgs


def join_data(imgs):
    by_type = list(zip(*imgs))
    scales = np.array(by_type[1])
    scales /= np.amin(scales)
    scaled_offsets = [s * np.array(offset) for s, offset in zip(scales, by_type[2])]
    ymin = [int(so[0]) for so in scaled_offsets]
    ymax = [int(img.shape[0] * s + so[0]) for img, so, s in zip(by_type[0], scaled_offsets, scales)]
    xmin = [int(so[1]) for so, s in zip(scaled_offsets, scales)]
    xmax = [int(img.shape[1] * s + so[1]) for img, so, s in zip(by_type[0], scaled_offsets, scales)]
    w, h = max(xmax) - min(xmin), max(ymax) - min(ymin)
    canvas = np.zeros((h, w, len(imgs)), dtype=np.float64)
    as_np = np.stack(scaled_offsets, axis=0)
    offset_shift = -np.amin(as_np, axis=0).astype(int)
    for i, img in enumerate(imgs):
        scaled_img = cv2.resize(img[0], (xmax[i] - xmin[i], ymax[i] - ymin[i]))
        canvas[ymin[i] + offset_shift[0]:ymax[i] + offset_shift[0], xmin[i] + offset_shift[0]:xmax[i] + offset_shift[1],
        i] = scaled_img
    return canvas


def viz(canvas):
    weight = 1. / canvas.shape[2]
    blended = np.zeros_like(canvas[:, :, 0])
    for i in range(canvas.shape[2]):
        blended += (weight * canvas[:, :, i]).astype(np.float64).astype(np.uint8)
    cv2.namedWindow('overlay', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('overlay', 600, 600)
    cv2.imshow("overlay", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_table(canvas, outfile, center_axes=False):
    dy = -int(canvas.shape[0] / 2) if center_axes else 0
    dx = -int(canvas.shape[1] / 2) if center_axes else 0
    with open(outfile, 'w', newline='') as csvfile:
        fieldnames = ['x', 'y'] + ["z_m" + str(i + 1) for i in range(canvas.shape[2])]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for y in range(canvas.shape[0]):
            for x in range(canvas.shape[1]):
                mydict = {**{"x": x + dx, "y": y + dy}, **dict(
                    zip(["z_m" + str(i + 1) for i in range(canvas.shape[2])], canvas[y, x, :].tolist()))}
                writer.writerow(mydict)


if __name__ == "__main__":
    offsets = {"ente_0.8": (0, 0, 0), "ente_1": (20, 50, 40),
               "ente_5": (-30, -30, -20)}  # in px of unscaled image and degree (third value). Change if given in metric units!
    parser = argparse.ArgumentParser("Align")
    parser.add_argument("img_dir", type=str, help="Folder with images or CSV tables")
    parser.add_argument("-d", "--show-debug", action="store_true")
    parser.add_argument("-n", "--no-center", action="store_false", help="don't center axes")
    parser.add_argument("-o", "--out", type=str)

    args = parser.parse_args()
    data = read_data(args.img_dir, offsets)
    joined = join_data(data)
    if args.out:
        write_table(joined, args.out, args.no_center)
    if (args.show_debug):
        viz(joined)
