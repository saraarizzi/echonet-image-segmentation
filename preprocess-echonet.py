from datetime import datetime
import os
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import collections as mc


VIDEOS_PATH = "data/echonet-dynamic/Videos"
TRACINGS_FILE = "data/echonet-dynamic/VolumeTracings.csv"
SPLITS_FILE = "data/echonet-dynamic/FileList.csv"


def prepare_dirs():
    os.makedirs(os.path.join(*["data", "echonet-processed", "train", "image"]), exist_ok=True)
    os.makedirs(os.path.join(*["data", "echonet-processed", "train", "mask"]), exist_ok=True)
    os.makedirs(os.path.join(*["data", "echonet-processed", "test", "image"]), exist_ok=True)
    os.makedirs(os.path.join(*["data", "echonet-processed", "test", "mask"]), exist_ok=True)
    os.makedirs(os.path.join(*["data", "echonet-processed", "val", "image"]), exist_ok=True)
    os.makedirs(os.path.join(*["data", "echonet-processed", "val", "mask"]), exist_ok=True)
    global DATA_PATH
    DATA_PATH = os.path.join(*["data", "echonet-processed"])
    print("Directories setup done")


def save_frames(f, codes, data_split):

    for frame_code in codes:
        vid_cap = cv.VideoCapture(f"{VIDEOS_PATH}/{f}")
        success, _ = vid_cap.read()
        count = 0

        img = []
        while success:
            success, frame = vid_cap.read()
            if count == frame_code - 2:
                img = [vid_cap.read()[1] for i in range(5)]
                success = False
            count += 1

        # Denoise 3rd frame considering all the 5 frames
        dst = cv.fastNlMeansDenoisingMulti(img, 2, 5, None, 4, 7, 35)

        # Histogram Equalization (CLAHE)
        gray_dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray_dst)

        cv.imwrite(os.path.join(*[DATA_PATH, data_split, "image", f"{f[:-4]}_{frame_code}.png"]), cl1)


def save_masks(traces, f, codes, data_split):
    px = 1 / plt.rcParams['figure.dpi']

    for code in codes:
        # Create plot of lines collection
        fig, ax = plt.subplots(figsize=(112 * px, 112 * px))
        ex1 = traces[
            (traces.Frame == code) & (traces.FileName == f)
        ]
        lines = []
        for index, row in ex1.iterrows():
            (x1, y1) = row['X1'], -row['Y1']
            (x2, y2) = row['X2'], -row['Y2']
            point = [(x1, y1), (x2, y2)]
            lines.append(point)
        lc = mc.LineCollection(lines, linewidths=2)
        ax.set_xlim(0, 112)
        ax.set_ylim(-112, 0)
        ax.add_collection(lc)
        plt.axis("off")
        plt.savefig("temp_mask.png")

        mask = cv.imread(f"temp_mask.png", cv.IMREAD_GRAYSCALE)
        _, bin_inv = cv.threshold(mask, 127, 255, cv.THRESH_BINARY_INV)
        k = np.ones((2, 2), np.uint8)
        d_mask = cv.dilate(bin_inv, k, iterations=1)
        cv.imwrite(os.path.join(*[DATA_PATH, data_split, "mask", f"{f[:-4]}_{code}.png"]), d_mask)
        plt.close()


def save_frame_mask(traces, f, codes, data_split):
    save_frames(f, codes, data_split)
    save_masks(traces, f, codes, data_split)


if __name__ == "__main__":
    prepare_dirs()
    exceptions = []
    not_processed = []

    print(f"Start processing at {datetime.now()}")

    # Read Tracings
    tracings = pd.read_csv(TRACINGS_FILE)

    # Read Splits
    splits = pd.read_csv(SPLITS_FILE)

    # Get all videos file names
    all_videos = os.listdir(VIDEOS_PATH)

    # Loop over each video
    for idx, file_name in enumerate(all_videos):

        if idx % 250 == 0:
            print(f"Processing \t ............. \t {idx}/{len(all_videos)} \t {np.round((idx/len(all_videos))*100, 3)}")

        try:

            # Get dataset split
            split = splits[splits["FileName"] == file_name[:-4]]["Split"].item()

            # Get frame codes
            frame_codes = list(tracings[tracings["FileName"] == file_name]["Frame"].unique())

            # Save frames and masks files
            save_frame_mask(tracings, file_name, frame_codes, split)

        except Exception as e:
            not_processed.append(file_name)
            exceptions.append(e)

    print(f"End processing at {datetime.now()}")

    # Check not processed videos
    print(f"Not processed: {len(not_processed)}")
    d = {"FileName": not_processed, "Exception": exceptions}
    df = pd.DataFrame(d)
    df.to_csv(os.path.join(*[DATA_PATH, "not_processed.csv"]))

    print("Done :)")
