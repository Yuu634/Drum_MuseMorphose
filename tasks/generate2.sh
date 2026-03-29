echo "ドラム譜面生成（難易度レベル指定）"

# 入力譜面の難易度をシフトして生成
python -m scripts.generate.generate_drum_from_input \
    --model_path trained_model/checkpoints_drum_cp_with_difficulty/params/step_130000-RC_0.217-KL_-0.000-model.pt \
    --input_midi data/test_dataset/Rock3.mid \
    --output_midi output/Rock3_130000step_cp.mid \
    --difficulty_shift 0 0 0 0 0