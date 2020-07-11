import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

im = np.array(Image.open('data/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/Potsdam_2_10_RGB.0.2.jpg'), dtype=np.uint8)
ann = open('data/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/Potsdam_2_10_RGB.0.2.txt').read().split("\n")


# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

# Create a Rectangle patch
for i in ann:
    if not i: continue
    i = [float(j)*256 for j in i.split(" ")]
    left = i[1]-i[3]/2
    bottom = i[2]-i[4]/2
    rect = patches.Rectangle((left, bottom),i[3],i[4],linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

plt.show()
