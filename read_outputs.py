import os
import csv
import glob
import argparse


def read_eval_csv(path_or_dir: str = "outputs", latest: bool = True, max_rows: int = None):
    """Đọc file CSV kết quả trong thư mục outputs hoặc từ đường dẫn chỉ định.

    - Nếu `path_or_dir` là thư mục: tìm file `eval_results_*.csv` mới nhất (khi latest=True).
    - Nếu `path_or_dir` là file: đọc trực tiếp.
    - Trả về (records, csv_path).
    """
    csv_path = path_or_dir
    if os.path.isdir(path_or_dir):
        pattern = os.path.join(path_or_dir, "eval_results_*.csv")
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(f"Không tìm thấy file CSV theo mẫu: {pattern}")
        csv_path = max(candidates, key=os.path.getmtime) if latest else sorted(candidates)[-1]

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Không tồn tại file CSV: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            records.append({
                "id": int(row.get("id", i)) if str(row.get("id", "")).strip().isdigit() else i,
                "ground_truth": row.get("ground_truth", ""),
                "prediction": row.get("prediction", ""),
            })
            if max_rows is not None and len(records) >= max_rows:
                break

    return records, csv_path


def print_eval_samples(records, limit: int = 3):
    """In nhanh một số mẫu từ danh sách bản ghi CSV."""
    for rec in records[:max(0, limit)]:
        print(f"ID: {rec['id']}")
        print(f"GT: {rec['ground_truth']}")
        print(f"PR: {rec['prediction']}")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Đọc file CSV kết quả từ thư mục outputs")
    parser.add_argument("path", nargs="?", default="outputs", help="Đường dẫn thư mục hoặc file CSV")
    parser.add_argument("--limit", type=int, default=3, help="Số mẫu in nhanh")
    parser.add_argument("--max_rows", type=int, default=None, help="Giới hạn số dòng đọc")
    parser.add_argument("--oldest", action="store_true", help="Chọn file cũ nhất thay vì mới nhất")
    args = parser.parse_args()

    records, csv_path = read_eval_csv(args.path, latest=not args.oldest, max_rows=args.max_rows)
    print(f"Đang đọc từ: {csv_path}")
    print_eval_samples(records, limit=args.limit)


if __name__ == "__main__":
    main()


