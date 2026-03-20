#!/bin/bash
# クイックスタート: CPUでドラム譜MuseMorphose学習を実行

set -e  # エラーが発生したら停止

echo "========================================"
echo "ドラム譜MuseMorphose クイックスタート"
echo "========================================"

# 現在のディレクトリを確認
if [ ! -f "train_drum.py" ]; then
    echo "エラー: このスクリプトはMuseMorphoseディレクトリから実行してください"
    exit 1
fi

# 1. 設定ファイルを確認・修正
echo ""
echo "[1/4] 設定ファイルを確認中..."
if [ ! -f "config/drum_config.yaml" ]; then
    echo "エラー: config/drum_config.yaml が見つかりません"
    exit 1
fi

# CPUモードに設定されているか確認
if grep -q "device: 'cuda'" config/drum_config.yaml; then
    echo "警告: 設定ファイルがCUDAモードになっています"
    echo "CPUモードで実行するために一時的に上書きします"
    DEVICE_OVERRIDE="--device cpu"
else
    DEVICE_OVERRIDE=""
fi

# 2. データセットの確認
echo ""
echo "[2/4] データセットを確認中..."
DATA_DIR=$(grep "data_dir:" config/drum_config.yaml | awk '{print $2}' | tr -d "'")
VOCAB_PATH=$(grep "vocab_path:" config/drum_config.yaml | awk '{print $2}' | tr -d "'")

if [ ! -d "$DATA_DIR" ]; then
    echo "エラー: データディレクトリが見つかりません: $DATA_DIR"
    echo ""
    echo "データセットを準備してください:"
    echo "  python prepare_drum_dataset.py \\"
    echo "      --midi_dir /path/to/your/drum/midis \\"
    echo "      --output_dir $DATA_DIR \\"
    echo "      --vocab_path $VOCAB_PATH"
    exit 1
fi

if [ ! -f "$VOCAB_PATH" ]; then
    echo "エラー: 語彙ファイルが見つかりません: $VOCAB_PATH"
    exit 1
fi

TRAIN_SPLIT="${DATA_DIR}/train_split.pkl"
VAL_SPLIT="${DATA_DIR}/val_split.pkl"

if [ ! -f "$TRAIN_SPLIT" ] || [ ! -f "$VAL_SPLIT" ]; then
    echo "エラー: 訓練/検証分割ファイルが見つかりません"
    echo "prepare_drum_dataset.pyでデータセットを準備してください"
    exit 1
fi

echo "✓ データセット確認完了"
echo "  データディレクトリ: $DATA_DIR"
echo "  語彙ファイル: $VOCAB_PATH"

# 3. 依存関係の確認
echo ""
echo "[3/4] 依存関係を確認中..."

python -c "
import sys
try:
    import torch
    import yaml
    import numpy as np
    from drum_tokenizer import DrumTokenizer
    from drum_dataloader import DrumTransformerDataset
    print('✓ すべての依存関係が利用可能です')
except ImportError as e:
    print(f'✗ 依存関係が不足しています: {e}')
    print('pip install -r requirements_drum.txt を実行してください')
    sys.exit(1)
" || exit 1

# 4. 学習を開始
echo ""
echo "[4/4] 学習を開始します..."
echo ""
echo "========================================"
echo "設定情報"
echo "========================================"
echo "  デバイス: CPU"
echo "  設定ファイル: config/drum_config.yaml"
echo "  データディレクトリ: $DATA_DIR"
echo ""
echo "学習を中断するには Ctrl+C を押してください"
echo "========================================"
echo ""

# 学習を実行
python train_drum.py --config config/drum_config.yaml $DEVICE_OVERRIDE

echo ""
echo "========================================"
echo "学習が完了しました！"
echo "========================================"
echo ""
echo "チェックポイントは以下に保存されています:"
CKPT_DIR=$(grep "ckpt_dir:" config/drum_config.yaml | awk '{print $2}' | tr -d "'")
echo "  $CKPT_DIR"
echo ""
echo "ログファイル:"
echo "  ${CKPT_DIR}/log.txt"
echo "  ${CKPT_DIR}/valloss.txt"
