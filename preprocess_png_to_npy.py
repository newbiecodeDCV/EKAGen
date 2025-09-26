import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def png_to_npy(png_path, out_path, size=(300, 300)):
    img = Image.open(png_path).convert("L")
    img = img.resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    np.save(out_path, arr)

def main():
    dataset_dir = "dataset/iu_xray"
    ann_path = os.path.join(dataset_dir, "annotation.json")
    out_ann_path = os.path.join(dataset_dir, "annotation_npy.json")
    out_img_root = os.path.join(dataset_dir, "images300_array")
    ensure_dir(out_img_root)

    with open(ann_path, "r") as f:
        data = json.load(f)

    new_data = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        for rec in tqdm(data[split], desc=f"Converting {split}"):
            case_id = rec["id"]
            views = rec["image_path"]
            report = rec["report"]

            case_dir = os.path.join(out_img_root, case_id)
            ensure_dir(case_dir)

            new_views = []
            for idx, v in enumerate(views):
                png_path = os.path.join(dataset_dir, v)
                out_path = os.path.join(case_dir, f"{idx}.npy")
                png_to_npy(png_path, out_path)
                new_views.append(os.path.relpath(out_path, dataset_dir))

            new_data[split].append({
                "id": case_id,
                "views": new_views,
                "report": report
            })

    with open(out_ann_path, "w") as fw:
        json.dump(new_data, fw, indent=2)

    print(f"Done. Saved new annotation with splits: {out_ann_path}")

if __name__ == "__main__":
    main()
