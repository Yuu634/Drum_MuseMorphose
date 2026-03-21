#!/usr/bin/env python3
import sys
import pickle
from pathlib import Path
from pprint import pprint


def main():
    if len(sys.argv) < 2:
        print("使用方法: python test.py <pkl_file_path>")
        sys.exit(1)

    pkl_path = sys.argv[1]
    path = Path(pkl_path)

    if not path.exists():
        print(f"エラー: ファイルが見つかりません: {path}")
        sys.exit(1)

    print(f"=== pklファイル内容: {path} ===\n")

    with path.open("rb") as f:
        obj = pickle.load(f)

    # vocab以外を出力
    if isinstance(obj, dict) and 'vocab' in obj:
        obj_filtered = {k: v for k, v in obj.items() if k != 'vocab'}
        pprint(obj_filtered)
    else:
        pprint(obj)


if __name__ == "__main__":
    main()