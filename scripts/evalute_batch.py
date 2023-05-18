import sys

sys.path.append("..")

from foodseg.evaluate import Evaluator

sys.path.remove("..")

from pathlib import Path


import cv2


pth_file = "pretrained/model_v1.0_epoch110.pth"
image_dir = "dataset-compressed/images"
output_dir = "output"

Path(output_dir).mkdir(exist_ok=True)

evalutor = Evaluator(pth_file)

for item in Path(image_dir).iterdir():
    output = evalutor.evaluate(cv2.imread(str(item)), timeit=True)
    cv2.imwrite(str(Path(output_dir) / item.name), output)
