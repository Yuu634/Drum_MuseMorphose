#!/usr/bin/env python
"""
ドラム譜トークンとMIDI変換のテストスクリプト
依存関係がインストールされているか確認し、基本的な動作をテストします
"""
import sys
import os

def test_imports():
    """必要なモジュールのインポートをテスト"""
    print("=" * 60)
    print("依存関係のチェック")
    print("=" * 60)

    modules = {
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'yaml': 'PyYAML',
        'miditoolkit': 'miditoolkit',
        'tqdm': 'tqdm'
    }

    missing = []
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - 未インストール")
            missing.append(name)

    if missing:
        print(f"\n警告: {', '.join(missing)} がインストールされていません")
        print("pip install -r requirements_drum.txt を実行してください")
        return False

    print("\n✓ すべての依存関係がインストールされています")
    return True


def test_tokenizer():
    """トークナイザーのテスト"""
    print("\n" + "=" * 60)
    print("トークナイザーのテスト")
    print("=" * 60)

    try:
        from drum_tokenizer import DrumTokenizer

        tokenizer = DrumTokenizer()
        print(f"✓ トークナイザーの初期化成功")
        print(f"  語彙サイズ: {tokenizer.vocab_size}")

        # トークンの例をいくつか表示
        print(f"\n  構造トークンの例:")
        structure_tokens = [t for t in tokenizer.vocab if t.startswith('<')][:10]
        for token in structure_tokens:
            print(f"    - {token}")

        print(f"\n  ドラムトークンの例:")
        drum_tokens = [t for t in tokenizer.vocab if not t.startswith('<')][:10]
        for token in drum_tokens:
            print(f"    - {token}")

        # 語彙を保存
        vocab_path = '/tmp/test_drum_vocab.pkl'
        tokenizer.save_vocab(vocab_path)
        print(f"\n✓ テスト語彙ファイルを保存: {vocab_path}")

        return True

    except Exception as e:
        print(f"✗ トークナイザーのテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_token_to_midi():
    """トークンからMIDIへの変換をテスト"""
    print("\n" + "=" * 60)
    print("Token to MIDI 変換のテスト")
    print("=" * 60)

    try:
        from drum_to_midi import tokens_to_midi

        # テスト用のトークン列
        test_tokens = [
            '<BAR>',
            '<BEAT_1>', '<POS_0>', 'KICK_HIT_Normal', 'HH_CLOSED_HIT_Normal',
            '<POS_6>', 'HH_CLOSED_HIT_Ghost',
            '<POS_12>', 'SNARE_HIT_Normal', 'HH_CLOSED_HIT_Normal',
            '<POS_18>', 'HH_CLOSED_HIT_Ghost',
            '<BEAT_2>', '<POS_0>', 'KICK_HIT_Accent', 'HH_CLOSED_HIT_Normal',
            '<POS_6>', 'HH_CLOSED_HIT_Ghost',
            '<POS_12>', 'SNARE_HIT_Accent', 'HH_CLOSED_HIT_Normal',
            '<POS_18>', 'HH_CLOSED_HIT_Ghost',
            '<BAR>',
            '<BEAT_1>', '<POS_0>', 'KICK_HIT_Normal', 'CRASH_HIT_Accent',
            '<POS_12>', 'SNARE_FLAM_Normal',
            '<BEAT_2>', '<POS_0>', 'KICK_HIT_Normal',
            '<POS_12>', 'SNARE_HIT_Normal',
            '<EOS>'
        ]

        output_path = '/tmp/test_drum_output.mid'
        midi_obj = tokens_to_midi(test_tokens, output_path, bpm=120)

        print(f"✓ MIDIファイルの生成成功")
        print(f"  出力: {output_path}")
        print(f"  ノート数: {len(midi_obj.instruments[0].notes)}")
        print(f"  BPM: 120")

        return True

    except Exception as e:
        print(f"✗ Token to MIDI 変換のテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_imports():
    """データローダーのインポートをテスト"""
    print("\n" + "=" * 60)
    print("データローダーのインポートテスト")
    print("=" * 60)

    try:
        from drum_dataloader import DrumTransformerDataset
        print(f"✓ データローダーのインポート成功")
        return True

    except Exception as e:
        print(f"✗ データローダーのインポート失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト関数"""
    print("\nドラム譜MuseMorphose - システムテスト\n")

    results = []

    # 依存関係のチェック
    results.append(("依存関係", test_imports()))

    # トークナイザーのテスト
    results.append(("トークナイザー", test_tokenizer()))

    # Token to MIDI 変換のテスト
    results.append(("Token→MIDI変換", test_token_to_midi()))

    # データローダーのインポートテスト
    results.append(("データローダー", test_dataloader_imports()))

    # 結果のサマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)

    for name, result in results:
        status = "✓ 成功" if result else "✗ 失敗"
        print(f"{name:20s}: {status}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n✓ すべてのテストが成功しました!")
        print("\n次は以下のコマンドでデータセットを準備してください:")
        print("  python prepare_drum_dataset.py --midi_dir /path/to/midis --output_dir ./drum_dataset --vocab_path ./drum_vocab.pkl")
        return 0
    else:
        print("\n✗ いくつかのテストが失敗しました")
        print("\n依存関係がすべてインストールされているか確認してください:")
        print("  pip install -r requirements_drum.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
