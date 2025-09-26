import os
import json

def main():
    dataset_dir = "dataset/iu_xray"
    ann_path = os.path.join(dataset_dir, "annotation.json")
    out_ann_path = os.path.join(dataset_dir, "annotation_png_fixed.json")

    with open(ann_path, "r") as f:
        data = json.load(f)

    new_data = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        for rec in data[split]:
            case_id = rec["id"]
            views = rec["image_path"]
            report = rec["report"]

            # Đảm bảo tất cả path đều đúng: images/CXRxxxx/0.png
            fixed_views = []
            for v in views:
                if not v.startswith("images/"):
                    v = os.path.join("images", v)
                fixed_views.append(v)

            new_data[split].append({
                "id": case_id,
                "image_path": fixed_views,
                "report": report
            })

    with open(out_ann_path, "w") as fw:
        json.dump(new_data, fw, indent=2)

    print(f"Annotation fixed. Saved: {out_ann_path}")

if __name__ == "__main__":
    main()
