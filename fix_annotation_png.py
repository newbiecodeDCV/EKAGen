import os
import json

def main():
    dataset_dir = "dataset/iu_xray"
    ann_in = os.path.join(dataset_dir, "annotation.json")       # annotation gốc
    ann_out = os.path.join(dataset_dir, "annotation_fixed.json")  # annotation mới

    with open(ann_in, "r") as f:
        data = json.load(f)

    new_data = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        for rec in data[split]:
            case_id = rec["id"]
            report = rec["report"]

            fixed_views = []
            for v in rec["image_path"]:
                # đảm bảo prefix là images/...
                if not v.startswith("images/"):
                    v = os.path.join("images", v)
                fixed_views.append(v)

            new_data[split].append({
                "id": case_id,
                "image_path": fixed_views,
                "report": report
            })

    with open(ann_out, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"Annotation fixed and saved to {ann_out}")

if __name__ == "__main__":
    main()
