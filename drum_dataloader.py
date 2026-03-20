"""
ドラム譜用データローダー
"""
import os
import pickle
import random
from glob import glob
from typing import List, Optional

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def pickle_load(path):
    """Pickleファイルを読み込み"""
    return pickle.load(open(path, 'rb'))


class DrumTransformerDataset(Dataset):
    """ドラム譜用のTransformerデータセット"""

    def __init__(
        self,
        data_dir: str,
        vocab_path: str,
        model_enc_seqlen: int = 128,
        model_dec_seqlen: int = 1280,
        model_max_bars: int = 16,
        pieces: Optional[List[str]] = None,
        pad_to_same: bool = True,
        appoint_st_bar: Optional[int] = None,
        dec_end_pad_value: Optional[str] = None
    ):
        """
        Args:
            data_dir: データディレクトリ
            vocab_path: 語彙ファイルパス
            model_enc_seqlen: エンコーダーの最大系列長
            model_dec_seqlen: デコーダーの最大系列長
            model_max_bars: 最大小節数
            pieces: 使用するデータファイルのリスト（Noneの場合は全て）
            pad_to_same: 系列を同じ長さにパディングするか
            appoint_st_bar: 開始小節を指定（Noneの場合はランダム）
            dec_end_pad_value: デコーダー末尾のパディング値（'EOS' or None）
        """
        self.vocab_path = vocab_path
        self.read_vocab()

        self.data_dir = data_dir
        self.pieces = pieces
        self.build_dataset()

        self.model_enc_seqlen = model_enc_seqlen
        self.model_dec_seqlen = model_dec_seqlen
        self.model_max_bars = model_max_bars

        self.pad_to_same = pad_to_same
        self.appoint_st_bar = appoint_st_bar

        if dec_end_pad_value is None:
            self.dec_end_pad_value = self.pad_token
        elif dec_end_pad_value == 'EOS':
            self.dec_end_pad_value = self.eos_token
        else:
            self.dec_end_pad_value = self.pad_token

    def read_vocab(self):
        """語彙を読み込み"""
        token2idx, idx2token = pickle_load(self.vocab_path)
        self.token2idx = token2idx
        self.idx2token = idx2token

        # 特殊トークン
        self.bar_token = self.token2idx['<BAR>']
        self.eos_token = self.token2idx['<EOS>']
        self.pad_token = len(self.token2idx)  # PADトークンは語彙の末尾に追加
        self.vocab_size = self.pad_token + 1

    def build_dataset(self):
        """データセットを構築"""
        if not self.pieces:
            self.pieces = sorted(glob(os.path.join(self.data_dir, '*.pkl')))
        else:
            self.pieces = sorted([os.path.join(self.data_dir, p) for p in self.pieces])

        # 統計ファイルなどを除外
        self.pieces = [p for p in self.pieces if
                       not os.path.basename(p).startswith('dataset_stats') and
                       not os.path.basename(p).startswith('train_split') and
                       not os.path.basename(p).startswith('val_split') and
                       not os.path.basename(p).startswith('failed_files')]

        self.piece_bar_pos = []

        for i, p in enumerate(self.pieces):
            if not i % 200:
                print(f'[preparing data] now at #{i}')

            try:
                bar_pos, p_evs = pickle_load(p)

                # 最後の小節位置を調整
                if bar_pos[-1] == len(p_evs):
                    bar_pos = bar_pos[:-1]

                if len(p_evs) - bar_pos[-1] <= 2:
                    # 空の末尾小節を除く
                    bar_pos = bar_pos[:-1]

                bar_pos.append(len(p_evs))
                self.piece_bar_pos.append(bar_pos)

            except Exception as e:
                print(f'Error loading {p}: {e}')
                self.piece_bar_pos.append([0])

    def get_sample_from_file(self, piece_idx: int):
        """ファイルからサンプルを取得"""
        piece_evs = pickle_load(self.pieces[piece_idx])[1]

        # 開始小節を選択
        if len(self.piece_bar_pos[piece_idx]) > self.model_max_bars and self.appoint_st_bar is None:
            picked_st_bar = random.choice(
                range(len(self.piece_bar_pos[piece_idx]) - self.model_max_bars)
            )
        elif self.appoint_st_bar is not None and \
             self.appoint_st_bar < len(self.piece_bar_pos[piece_idx]) - self.model_max_bars:
            picked_st_bar = self.appoint_st_bar
        else:
            picked_st_bar = 0

        piece_bar_pos = self.piece_bar_pos[piece_idx]

        # 小節数に応じてトリミングまたはパディング
        if len(piece_bar_pos) > self.model_max_bars:
            piece_evs = piece_evs[
                piece_bar_pos[picked_st_bar]: piece_bar_pos[picked_st_bar + self.model_max_bars]
            ]
            picked_bar_pos = np.array(
                piece_bar_pos[picked_st_bar: picked_st_bar + self.model_max_bars]
            ) - piece_bar_pos[picked_st_bar]
            n_bars = self.model_max_bars
        else:
            picked_bar_pos = np.array(
                piece_bar_pos + [piece_bar_pos[-1]] * (self.model_max_bars - len(piece_bar_pos))
            )
            n_bars = len(piece_bar_pos)
            assert len(picked_bar_pos) == self.model_max_bars

        return piece_evs, picked_st_bar, picked_bar_pos, n_bars

    def pad_sequence(self, seq: List[int], maxlen: int, pad_value: Optional[int] = None) -> List[int]:
        """系列をパディング"""
        if pad_value is None:
            pad_value = self.pad_token

        seq.extend([pad_value for _ in range(maxlen - len(seq))])
        return seq

    def get_encoder_input_data(self, bar_positions: np.ndarray, bar_tokens: List[int]):
        """エンコーダー入力データを生成"""
        assert len(bar_positions) == self.model_max_bars + 1

        enc_padding_mask = np.ones((self.model_max_bars, self.model_enc_seqlen), dtype=bool)
        enc_padding_mask[:, :2] = False
        padded_enc_input = np.full(
            (self.model_max_bars, self.model_enc_seqlen),
            dtype=int,
            fill_value=self.pad_token
        )
        enc_lens = np.zeros((self.model_max_bars,))

        for b, (st, ed) in enumerate(zip(bar_positions[:-1], bar_positions[1:])):
            enc_padding_mask[b, :(ed - st)] = False
            enc_lens[b] = ed - st
            within_bar_tokens = self.pad_sequence(
                bar_tokens[st:ed].copy() if isinstance(bar_tokens, np.ndarray) else bar_tokens[st:ed],
                self.model_enc_seqlen,
                self.pad_token
            )
            within_bar_tokens = np.array(within_bar_tokens)

            padded_enc_input[b, :] = within_bar_tokens[:self.model_enc_seqlen]

        return padded_enc_input, enc_padding_mask, enc_lens

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        """データセットから1サンプルを取得"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        bar_tokens, st_bar, bar_pos, enc_n_bars = self.get_sample_from_file(idx)

        # トークンのリストに変換
        if isinstance(bar_tokens, np.ndarray):
            bar_tokens = bar_tokens.tolist()

        bar_pos = bar_pos.tolist() + [len(bar_tokens)]

        # エンコーダー入力データを生成
        enc_inp, enc_padding_mask, enc_lens = self.get_encoder_input_data(
            np.array(bar_pos), bar_tokens
        )

        # デコーダー入力/ターゲットを生成
        length = len(bar_tokens)
        if self.pad_to_same:
            inp = self.pad_sequence(bar_tokens, self.model_dec_seqlen + 1)
        else:
            inp = self.pad_sequence(
                bar_tokens,
                len(bar_tokens) + 1,
                pad_value=self.dec_end_pad_value
            )

        target = np.array(inp[1:], dtype=int)
        inp = np.array(inp[:-1], dtype=int)
        assert len(inp) == len(target)

        return {
            'id': idx,
            'piece_id': int(os.path.basename(self.pieces[idx]).replace('.pkl', '')),
            'st_bar_id': st_bar,
            'bar_pos': np.array(bar_pos, dtype=int),
            'enc_input': enc_inp,
            'dec_input': inp[:self.model_dec_seqlen],
            'dec_target': target[:self.model_dec_seqlen],
            'length': min(length, self.model_dec_seqlen),
            'enc_padding_mask': enc_padding_mask,
            'enc_length': enc_lens,
            'enc_n_bars': enc_n_bars
        }


def collate_fn(batch):
    """カスタムコレート関数（バッチ処理用）"""
    return {
        key: torch.tensor([item[key] for item in batch]) if key in batch[0] else None
        for key in batch[0].keys()
    }


if __name__ == "__main__":
    # テスト用
    print("Testing DrumTransformerDataset...")

    # ダミーの語彙ファイルを作成
    test_vocab_path = '/tmp/test_drum_vocab.pkl'
    if not os.path.exists(test_vocab_path):
        from drum_tokenizer import DrumTokenizer
        tokenizer = DrumTokenizer()
        tokenizer.save_vocab(test_vocab_path)
        print(f"Created test vocabulary: {test_vocab_path}")

    # データセットディレクトリを指定（実際のデータがある場合）
    data_dir = './drum_dataset'

    if os.path.exists(data_dir):
        dset = DrumTransformerDataset(
            data_dir,
            test_vocab_path,
            model_max_bars=16,
            model_dec_seqlen=1280,
            model_enc_seqlen=128,
            pad_to_same=True
        )

        print(f'Vocabulary size: {dset.vocab_size}')
        print(f'Bar token: {dset.bar_token}')
        print(f'Pad token: {dset.pad_token}')
        print(f'Dataset length: {len(dset)}')

        if len(dset) > 0:
            sample = dset[0]
            print(f'\nSample keys: {sample.keys()}')
            print(f'Encoder input shape: {sample["enc_input"].shape}')
            print(f'Decoder input shape: {sample["dec_input"].shape}')
            print(f'Decoder target shape: {sample["dec_target"].shape}')

            # DataLoaderのテスト
            dloader = DataLoader(dset, batch_size=2, shuffle=False, num_workers=0)
            batch = next(iter(dloader))
            print(f'\nBatch keys: {batch.keys()}')
            for k, v in batch.items():
                if torch.is_tensor(v):
                    print(f'  {k}: {v.dtype}, {v.size()}')
    else:
        print(f"Data directory '{data_dir}' not found. Please prepare dataset first.")
