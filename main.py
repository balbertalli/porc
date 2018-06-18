"""Run PORC on collection of impulses."""

import os
from porc import roomcomp

impulse_dir = 'impulses'
impulse_filenames = os.listdir(impulse_dir)

for filename in impulse_filenames:
    if "compensated" not in filename and filename.endswith(".wav"):
        print("Processing {}".format(filename))
        path = os.path.join(impulse_dir, filename)
        output = os.path.join(impulse_dir, filename[:-4] + "_compensated.wav")
        roomcomp(path, output, "data/target/44.1kHz/subultra-44.1.txt", None, True, 'wav', True, 0.05, True)
