echo "=== ドラム譜面データセットの難易度ラベル付与 ==="

# 変数
DRUM_DATASET_DIR="data/processed/drum_dataset_remi" # 難易度ラベル付与前データセットパス
DRUM_VOCAB_PATH="data/metadata/drum_vocab_remi.pkl" # ドラム譜面の語彙ファイルパス
DIFFICULTY_BOUNDS_PATH="data/metadata/difficulty_bounds_remi.pkl" # 難易度の境界値を保存するファイルパス
DIFFICULTY_DATASET_DIR="data/processed/drum_dataset_remi_with_difficulty" # 難易度ラベル付与後データセットパス
DEFAULT_BPM=120 # BPMのデフォルト値

# 各難易度項目の境界値を計算
python -m src.utils.compute_difficulty_bounds \
    ${DRUM_DATASET_DIR} \
    ${DRUM_VOCAB_PATH} \
    ${DIFFICULTY_BOUNDS_PATH} \
    ${DEFAULT_BPM}

# ドラム譜面データセットに難易度ラベルを追加する
python -m scripts.preprocess.prepare_drum_dataset_with_difficulty \
    ${DRUM_DATASET_DIR} \
    ${DRUM_VOCAB_PATH} \
    ${DIFFICULTY_BOUNDS_PATH} \
    ${DIFFICULTY_DATASET_DIR} \
    ${DEFAULT_BPM}