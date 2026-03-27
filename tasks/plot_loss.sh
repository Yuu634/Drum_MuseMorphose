echo "学習曲線作成スクリプト"

# train-loss変数
LOG_PATH="trained_model/checkpoints_drum_cp_with_difficulty/log.txt"
LOSS_OUTPUT_DIR="logs/cp"

# val-loss変数
VALLOG_PATH="trained_model/checkpoints_drum_cp_with_difficulty/valloss.txt"
VALLOSS_OUTPUT_DIR="logs/cp"


python src/utils/plot_loss.py --log "$LOG_PATH" --output "$LOSS_OUTPUT_DIR"
python src/utils/plot_valloss.py --log "$VALLOG_PATH" --output "$VALLOSS_OUTPUT_DIR"