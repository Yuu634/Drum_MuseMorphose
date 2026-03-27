#!/usr/bin/env python3
import sys
import pickle
from pathlib import Path
from pprint import pprint


def main():
    if len(sys.argv) < 2:
        print("使用方法: python print_pkl.py <pkl_file_path> [bar_count]")
        print("  pkl_file_path: 対象のpklファイルパス")
        print("  bar_count: 表示する小節数（オプション。指定時は最初のN小節のみ表示）")
        sys.exit(1)

    pkl_path = sys.argv[1]
    bar_count = None
    
    if len(sys.argv) >= 3:
        try:
            bar_count = int(sys.argv[2])
        except ValueError:
            print(f"エラー: bar_countは整数で指定してください: {sys.argv[2]}")
            sys.exit(1)
    
    path = Path(pkl_path)

    if not path.exists():
        print(f"エラー: ファイルが見つかりません: {path}")
        sys.exit(1)

    print(f"=== pklファイル内容: {path} ===\n")

    with path.open("rb") as f:
        obj = pickle.load(f)

    # bar_count が指定されている場合、指定した小節数分のみを抽出
    if bar_count is not None and isinstance(obj, dict) and 'bar_positions' in obj and 'tokens' in obj:
        bar_positions = obj['bar_positions']
        tokens = obj['tokens']
        
        # 指定された小節数が有効な範囲か確認
        if bar_count < 1:
            print(f"エラー: bar_count は1以上である必要があります")
            sys.exit(1)
        
        if bar_count > len(bar_positions):
            print(f"警告: 小節数が多すぎます。総小節数は{len(bar_positions)}です。\n")
            end_idx = len(tokens)
        else:
            # bar_count 番目の次の小節の開始位置、または末尾
            if bar_count < len(bar_positions):
                end_idx = bar_positions[bar_count]
            else:
                end_idx = len(tokens)
        
        # 抽出したデータを作成
        extracted_tokens = tokens[:end_idx]
        extracted_bar_positions = bar_positions[:bar_count]
        
        obj_filtered = {
            'tokens': extracted_tokens,
            'bar_positions': extracted_bar_positions,
            'bar_count': bar_count,
            'total_tokens': len(extracted_tokens)
        }
        
        if 'indices' in obj:
            obj_filtered['indices'] = obj['indices'][:end_idx]
        
        print(f"【最初の {bar_count} 小節のみを表示】\n")
        pprint(obj_filtered)
    else:
        # vocab以外を出力
        if isinstance(obj, dict) and 'vocab' in obj:
            obj_filtered = {k: v for k, v in obj.items() if k != 'vocab'}
            pprint(obj_filtered)
        else:
            pprint(obj)
    
    # トークン種類数を表示
    if isinstance(obj, dict) and 'vocab' in obj:
        pprint(f"【トークン種類数】: {len(obj['vocab'])}")
    elif isinstance(obj, tuple) and len(obj) >= 2:
        # Tupleの場合、Dict要素の長さを取得
        vocab_candidate = None
        for item in obj:
            if isinstance(item, dict):
                vocab_candidate = item
                break
        if vocab_candidate is not None:
            pprint(f"【トークン種類数】: {len(vocab_candidate)}")
        else:
            pprint("【トークン種類数】: 不明")
    else:
        pprint("【トークン種類数】: 不明")


if __name__ == "__main__":
    main()