"""
データセット全体から難易度指標の境界値を計算

使用方法:
    python compute_difficulty_bounds.py <data_dir> <vocab_path> <output_path>

例:
    python compute_difficulty_bounds.py ./drum_dataset ./drum_vocab.pkl ./difficulty_bounds.pkl
"""

import sys
import os

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from drum_difficulty_calculator import compute_difficulty_bounds


def main():
    if len(sys.argv) < 4:
        print("Usage: python compute_difficulty_bounds.py <data_dir> <vocab_path> <output_path> [bpm]")
        print("\nExample:")
        print("  python compute_difficulty_bounds.py ./drum_dataset ./drum_vocab.pkl ./difficulty_bounds.pkl 120")
        sys.exit(1)

    data_dir = sys.argv[1]
    vocab_path = sys.argv[2]
    output_path = sys.argv[3]
    bpm = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0

    # データディレクトリとvocabファイルの存在確認
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        sys.exit(1)

    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file '{vocab_path}' does not exist.")
        sys.exit(1)

    print(f"Computing difficulty bounds...")
    print(f"  Data directory: {data_dir}")
    print(f"  Vocabulary: {vocab_path}")
    print(f"  Output: {output_path}")
    print(f"  Default BPM: {bpm}")
    print()

    bounds = compute_difficulty_bounds(data_dir, vocab_path, output_path, bpm=bpm)

    print("\n" + "=" * 60)
    print("Difficulty bounds computed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
