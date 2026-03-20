# MuseMorphose

This repository contains the official implementation of the following paper:  

* Shih-Lun Wu, Yi-Hsuan Yang  
**_MuseMorphose_: Full-Song and Fine-Grained Piano Music Style Transfer with One Transformer VAE**  
accepted to _IEEE/ACM Trans. Audio, Speech, & Language Processing (TASLP)_, Dec 2022 [<a href="https://arxiv.org/abs/2105.04090" target="_blank">arXiv</a>] [<a href="https://slseanwu.github.io/site-musemorphose/" target="_blank">demo website</a>]

## Prerequisites
* Python >= 3.6
* Install dependencies
```bash
pip3 install -r requirements.txt
```
* GPU with >6GB RAM (optional, but recommended)

## Preprocessing
```bash
# download REMI-pop-1.7K dataset
wget -O remi_dataset.tar.gz https://zenodo.org/record/4782721/files/remi_dataset.tar.gz?download=1
tar xzvf remi_dataset.tar.gz
rm remi_dataset.tar.gz

# compute attributes classes
python3 attributes.py
```

## Training
```bash
python3 train.py [config file]
```
* e.g.
```bash
python3 train.py config/default.yaml
```
* Or, you may download the pretrained weights straight away
```bash
wget -O musemorphose_pretrained_weights.pt https://zenodo.org/record/5119525/files/musemorphose_pretrained_weights.pt?download=1
```

## Generation
```bash
python3 generate.py [config file] [ckpt path] [output dir] [num pieces] [num samples per piece]
```
* e.g.
```bash
python3 generate.py config/default.yaml musemorphose_pretrained_weights.pt generations/ 10 5
```

This script will randomly draw the specified # of pieces from the test set.  
For each sample of a piece, the _rhythmic intensity_ and _polyphonicity_ will be shifted entirely and randomly by \[-3, 3\] classes for the model to generate style-transferred music.  
You may modify `random_shift_attr_cls()` in `generate.py` or write your own function to set the attributes.

## Customized Generation (To Be Added)
We welcome the community's suggestions and contributions for an interface on which users may
 * upload their own MIDIs, and 
 * set their desired bar-level attributes easily

## Drum Workflow: Data Preparation, Token Design, and Training

This repository also includes a drum-specific pipeline that tokenizes drum MIDI files with explicit playing-technique tokens (e.g., flam, rimshot, choke) and trains MuseMorphose from scratch on that representation.

### 1. Install Drum Dependencies
```bash
pip3 install -r requirements_drum.txt
```

### 2. Prepare Drum Dataset from MIDI
Run the dataset preparation script to:
1. build and save a drum vocabulary (`drum_vocab.pkl`),
2. tokenize all MIDI files,
3. save tokenized samples as `*.pkl`, and
4. create `train_split.pkl` / `val_split.pkl`.

```bash
python3 prepare_drum_dataset.py \
    --midi_dir ./drum_dataset \
    --output_dir ./drum_prepare \
    --vocab_path ./drum_vocab.pkl \
    --train_ratio 0.9 \
    --seed 42 \
    --file_extension .midi
```

If your files use `.mid`, set `--file_extension .mid`.

### 3. Drum Token Design (What the Script Produces)
The tokenizer (`drum_tokenizer.py`) uses:
1. Structural tokens: `<BAR>`, `<BEAT_1>` ... `<BEAT_4>`, `<POS_0>` ... `<POS_23>`, `<EOS>`, `<PAD>`
2. Performance tokens: `[INSTRUMENT]_[TECHNIQUE]_[VELOCITY_LEVEL]` (e.g., `SNARE_HIT_Normal`, `SNARE_FLAM_Accent`, `CRASH_CHOKE`)
3. Velocity levels: `Ghost`, `Normal`, `Accent`

Each prepared sample `drum_prepare/xxxxxx.pkl` is saved as:
```python
(bar_positions, token_indices)
```
where:
1. `bar_positions` stores token start indices per bar,
2. `token_indices` is the token-id sequence for training.

### 4. Configure Drum Training
Edit `config/drum_config.yaml` as needed:
1. `data.data_dir: './drum_prepare'`
2. `data.vocab_path: './drum_vocab.pkl'`
3. `data.train_split: './drum_prepare/train_split.pkl'`
4. `data.val_split: './drum_prepare/val_split.pkl'`
5. `training.device: 'cpu'` or `'cuda'`

### 5. Start Training
```bash
python3 train_drum.py --config config/drum_config.yaml
```

Or run the CPU quick-start helper:
```bash
bash quickstart_cpu.sh
```

### 6. Outputs
Training checkpoints and logs are written to `checkpoints_drum/` (or your configured `ckpt_dir`):
1. `params/`: model checkpoints,
2. `optim/`: optimizer states,
3. `log.txt`: training log,
4. `valloss.txt`: validation log.

For more details, see `README_drum.md`.

## Citation BibTex
If you find this work helpful and use our code in your research, please kindly cite our paper:
```
@article{wu2023musemorphose,
    title={{MuseMorphose}: Full-Song and Fine-Grained Piano Music Style Transfer with One {Transformer VAE}},
    author={Shih-Lun Wu and Yi-Hsuan Yang},
    year={2023},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
}
```
