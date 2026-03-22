# MuseMorphose CP Mode Guide

This document describes the limb-based CP mode implemented for drum training and generation.

## Overview

CP mode introduces a composite event representation:

- CP event: [Position, Hand1, Hand2, Right_Foot, Left_Foot]
- Structural event: <TEMPO_xxx>, <BAR>, <BEAT_x>, <EOS>

Key points:

- Hand slots are hand-agnostic (Hand1/Hand2), not right/left hand fixed.
- HH_PEDAL is assigned to Left_Foot.
- Hi-hat hits can be assigned to Hand1/Hand2.
- CP element tokens keep instrument+technique+velocity information.

## Tokenization Method Switch

Use config key:

- data.tokenization_method: standard or cp_limb_v1

Files where this switch is supported:

- train_drum.py
- prepare_drum_dataset.py
- drum_dataloader.py

## CP Embedding Design

CP mode uses CP-specific embedding in the model:

1. Embed each CP element independently:
   - Position embedding
   - Hand1 embedding
   - Hand2 embedding
   - Right_Foot embedding
   - Left_Foot embedding
2. Concatenate the five embeddings
3. Project through a linear layer to model dimension

Config keys:

- model.d_cp_pos_emb
- model.d_cp_limb_emb

## CP Output Heads

CP mode predicts multi-head outputs per timestep:

- event_type head
- structural head
- cp_pos head
- cp_hand1 head
- cp_hand2 head
- cp_right_foot head
- cp_left_foot head

Loss is masked by event type:

- event_type: all valid timesteps
- structural: structural timesteps only
- CP element heads: CP timesteps only

Config key for per-head weighting:

- model.cp_loss_weights

## Data Preparation (CP)

### 1) Build CP dataset

Example:

python prepare_drum_dataset.py \
  --midi_dir /path/to/midis \
  --output_dir ./drum_dataset_cp \
  --vocab_path ./drum_vocab_cp.pkl \
  --file_extension .mid \
  --tokenization_method cp_limb_v1

### 2) (Optional) Add difficulty labels

Example:

python prepare_drum_dataset_with_difficulty.py \
  ./drum_dataset_cp \
  ./drum_vocab_cp.pkl \
  ./difficulty_bounds.pkl \
  ./drum_dataset_cp_with_difficulty

## Training (CP)

Set in config:

- data.tokenization_method: cp_limb_v1
- data.vocab_path: path to CP vocab pickle
- data.data_dir: CP dataset directory

Then run:

python train_drum.py --config config/drum_config_gpu.yaml

Notes:

- Existing standard mode remains supported.
- Checkpoint compatibility requires matching tokenization mode and head setup.

## CP Generation Script

New script:

- generate_drum_cp.py

Usage:

python generate_drum_cp.py <config> <ckpt> <output_dir> <n_pieces> <n_samples_per_piece>

Example:

python generate_drum_cp.py \
  configs/train_drum_difficulty.yaml \
  checkpoints_drum_gpu/params/step_50000-RC_0.123-KL_-0.000-model.pt \
  ./outputs_cp \
  3 \
  2

Output files:

- *.mid : generated MIDI
- *.txt : generated CP event log

## CP to MIDI Conversion Helpers

Added helper functions in drum_to_midi.py:

- cp_events_to_tokens(cp_data, idx2struct_token, idx2limb_token)
- cp_data_to_midi(cp_data, idx2struct_token, idx2limb_token, output_path, bpm=None)

These convert CP event sequences to existing token form and then to MIDI.

## Current Scope and Limitation

- CP training path is implemented.
- CP generation script is implemented.
- Existing generate_drum_difficulty.py is kept for standard-oriented generation and intentionally rejects cp_limb_v1 mode.

## Minimal Config Example (CP)

model:
  d_cp_pos_emb: 64
  d_cp_limb_emb: 64
  cp_loss_weights:
    event_type: 1.0
    structural: 1.0
    cp_pos: 1.0
    cp_hand1: 1.0
    cp_hand2: 1.0
    cp_right_foot: 1.0
    cp_left_foot: 1.0

data:
  tokenization_method: cp_limb_v1
  vocab_path: ./drum_vocab_cp.pkl
  data_dir: ./drum_dataset_cp
