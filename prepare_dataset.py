"""
Create a dataset compatible with the training, same format of ADE20K.
Single BW image, each value from 0 to 255 means where that class_id is present in the original image.
"""
import argparse
import csv
import os

from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gt_dir", required=True, type=str, help="GT directory")
    parser.add_argument("--input_format", required=False, type=str, default="png", help="Input image format")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for new ground truth")
    parser.add_argument("--output_format", required=False, type=str, default="png", help="Output image format")
    parser.add_argument("--classes_csv", required=True, type=str, help="CSV file with class-colors mapping")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    files = sorted(glob(args.gt_dir + "/*." + args.input_format))
    print("Input files: " + str(files))

    os.makedirs(args.output_dir, exist_ok=True)

    classes = []

    # Read csv mapping like fasseg_classes.csv
    with open(args.classes_csv) as csv_file:
        reader = csv.reader(csv_file, delimiter=";")
        next(reader)
        for row in reader:
            classes.append(row)

    print("Classes: " + str(classes))

    for single_class in classes:
        single_class[1] = np.array(single_class[1].split(",")).astype(np.uint8)

    # Create new ground truth images starting from zero PNG image and adding classes != background
    for file in tqdm(files):
        old_gt = cv2.imread(file)
        new_gt = np.zeros(shape=old_gt.shape[:2], dtype=np.uint8)

        # Ignore background and assign other classes
        for class_idx, single_class in enumerate(classes[1:]):
            mask = cv2.inRange(old_gt, single_class[1], single_class[1])
            new_gt[mask == 255] = (class_idx + 1)
            if args.debug:
                print(single_class[0])
                print(single_class[1])
                cv2.imshow("debug", mask)
                cv2.imshow("new_gt", new_gt)
                cv2.waitKey(0)

        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(file[:file.rfind(".") + 1]) + args.output_format),
                    new_gt)


if __name__ == "__main__":
    main()
