import os
import cv2
import numpy as np

ALPHA_FILE_ROOT   = "/media/lobst3rd/DATA/dataset/matting/train/gt_training_lowres"
FOREGND_FLIE_ROOT = "/media/lobst3rd/DATA/dataset/matting/train/input_training_lowres"
BACKGND_FILE_ROOT = "/media/lobst3rd/DATA/dataset/bg_flickr"

OUTPUT_ROOT = "/media/lobst3rd/DATA/dataset/matting/output"
if not os.path.exists(OUTPUT_ROOT):
    os.mkdir(OUTPUT_ROOT)

alpha_files   = [os.path.join(ALPHA_FILE_ROOT, f) for f in os.listdir(ALPHA_FILE_ROOT) if os.path.isfile(os.path.join(ALPHA_FILE_ROOT, f))]
foregnd_files = [os.path.join(FOREGND_FLIE_ROOT, f) for f in os.listdir(FOREGND_FLIE_ROOT) if os.path.isfile(os.path.join(FOREGND_FLIE_ROOT, f))]
backgnd_files = []
backgnd_files += [os.path.join(BACKGND_FILE_ROOT, "forest", f) for f in os.listdir(os.path.join(BACKGND_FILE_ROOT, "forest")) if os.path.isfile(os.path.join(BACKGND_FILE_ROOT, "forest", f))]
backgnd_files += [os.path.join(BACKGND_FILE_ROOT, "nature", f) for f in os.listdir(os.path.join(BACKGND_FILE_ROOT, "nature")) if os.path.isfile(os.path.join(BACKGND_FILE_ROOT, "nature", f))]
backgnd_files += [os.path.join(BACKGND_FILE_ROOT, "scene",  f) for f in os.listdir(os.path.join(BACKGND_FILE_ROOT, "scene")) if os.path.isfile(os.path.join(BACKGND_FILE_ROOT, "scene", f))]

count = 0

for idx, foregnd_file in enumerate(foregnd_files):
    if not (idx in [2, 7, 8]):
        continue

    print(str(idx+1))

    foreground = cv2.imread(foregnd_file)
    foreground = foreground.astype(float) # Convert uint8 to float

    alpha_file = alpha_files[idx]
    alpha = cv2.imread(alpha_file)
    alpha = alpha.astype(float) / 255.0 # Normalize the alpha mask to keep intensity between 0 and 1

    output_path = os.path.join(OUTPUT_ROOT, str(idx+1))
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for backgnd_file in backgnd_files:

        background = cv2.imread(backgnd_file)

        if foreground.shape[0] > foreground.shape[1] and background.shape[0] > background.shape[1]:

            background = cv2.resize(background, (foreground.shape[1], foreground.shape[0])) 
            background = background.astype(float)
            background = cv2.multiply(1.0 - alpha, background)
            
            outImage = cv2.add(foreground, background)

            cv2.imwrite(os.path.join(output_path, "%d.png" % (count)), outImage)
            count += 1
