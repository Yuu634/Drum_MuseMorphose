"""
ドラム譜トークナイザー
MIDIファイルをドラム譜特有のトークン表記法に変換する
"""
import numpy as np
import miditoolkit
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# 定数定義
DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
POSITIONS_PER_BEAT = 24  # 1拍を24分割

# GM Drum Map (General MIDI standard)
DRUM_NOTE_MAP = {
    # Kick
    35: ('KICK', 'HIT'),
    36: ('KICK', 'HIT'),

    # Snare
    38: ('SNARE', 'HIT'),
    40: ('SNARE', 'HIT'),
    37: ('SNARE', 'XSTICK'),  # Side stick

    # Tom
    41: ('TOM1', 'HIT'),  # Low floor tom
    43: ('TOM1', 'HIT'),  # High floor tom
    45: ('TOM2', 'HIT'),  # Low tom
    47: ('TOM2', 'HIT'),  # Low-mid tom
    48: ('TOM2', 'HIT'),  # Hi-mid tom
    50: ('TOM2', 'HIT'),  # High tom

    # Floor tom
    41: ('FLOOR', 'HIT'),
    43: ('FLOOR', 'HIT'),

    # Hi-hat
    42: ('HH_CLOSED', 'HIT'),  # Closed hi-hat
    44: ('HH_CLOSED', 'PEDAL'),  # Pedal hi-hat
    46: ('HH_OPEN', 'HIT'),  # Open hi-hat

    # Ride
    51: ('RIDE_BOW', 'HIT'),  # Ride cymbal 1
    59: ('RIDE_BOW', 'HIT'),  # Ride cymbal 2
    53: ('RIDE_BELL', 'HIT'),  # Ride bell

    # Crash
    49: ('CRASH', 'HIT'),  # Crash cymbal 1
    57: ('CRASH', 'HIT'),  # Crash cymbal 2

    # Other cymbals
    52: ('CHINA', 'HIT'),  # Chinese cymbal
    55: ('SPLASH', 'HIT'),  # Splash cymbal
}


class DrumTokenizer:
    """ドラム譜トークナイザー"""

    def __init__(self):
        self.vocab = self._build_vocabulary()
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.vocab_size = len(self.vocab)

    def _build_vocabulary(self) -> List[str]:
        """トークン語彙を構築"""
        vocab = []

        # 特殊トークン
        vocab.extend(['<PAD>', '<EOS>', '<BAR>'])

        # 拍トークン (4拍子のみサポート)
        vocab.extend([f'<BEAT_{i}>' for i in range(1, 5)])

        # 位置トークン (1拍を24分割)
        vocab.extend([f'<POS_{i}>' for i in range(24)])

        # スネアドラム
        for vel in ['Ghost', 'Normal', 'Accent']:
            vocab.append(f'SNARE_HIT_{vel}')
        for vel in ['Normal', 'Accent']:
            vocab.extend([
                f'SNARE_RIMSHOT_{vel}',
                f'SNARE_XSTICK_{vel}',
                f'SNARE_FLAM_{vel}'
            ])
        vocab.append('SNARE_ROLL')

        # キック
        for vel in ['Normal', 'Accent']:
            vocab.append(f'KICK_HIT_{vel}')

        # タム類
        for drum in ['TOM1', 'TOM2', 'FLOOR']:
            for vel in ['Normal', 'Accent']:
                vocab.extend([
                    f'{drum}_HIT_{vel}',
                    f'{drum}_FLAM_{vel}'
                ])

        # ハイハット
        for vel in ['Ghost', 'Normal', 'Accent']:
            vocab.append(f'HH_CLOSED_HIT_{vel}')
        for vel in ['Normal', 'Accent']:
            vocab.extend([
                f'HH_HALFOPEN_HIT_{vel}',
                f'HH_OPEN_HIT_{vel}'
            ])
        vocab.append('HH_PEDAL')

        # ライドシンバル
        for vel in ['Ghost', 'Normal', 'Accent']:
            vocab.append(f'RIDE_BOW_HIT_{vel}')
        for vel in ['Normal', 'Accent']:
            vocab.append(f'RIDE_BELL_HIT_{vel}')

        # クラッシュ・その他シンバル
        for cymbal in ['CRASH', 'SPLASH', 'CHINA']:
            for vel in ['Normal', 'Accent']:
                vocab.append(f'{cymbal}_HIT_{vel}')
            vocab.append(f'{cymbal}_CHOKE')

        return vocab

    def _velocity_to_level(self, velocity: int) -> str:
        """ベロシティをGhost/Normal/Accentに変換"""
        if velocity < 40:
            return 'Ghost'
        elif velocity < 100:
            return 'Normal'
        else:
            return 'Accent'

    def _detect_flam(self, notes: List[miditoolkit.Note], window_ticks: int = 60) -> Dict[int, List[int]]:
        """
        フラムを検出 (短い間隔で連続する同楽器の打撃)

        Returns:
            Dict[主音符のインデックス, 前打音のインデックスのリスト]
        """
        flams = defaultdict(list)
        notes_sorted = sorted(notes, key=lambda x: (x.start, x.pitch))

        for i in range(len(notes_sorted) - 1):
            curr_note = notes_sorted[i]
            next_note = notes_sorted[i + 1]

            # 同じ楽器で極めて短い間隔の場合、フラムとみなす
            if (curr_note.pitch in DRUM_NOTE_MAP and
                next_note.pitch in DRUM_NOTE_MAP):
                curr_drum = DRUM_NOTE_MAP[curr_note.pitch][0]
                next_drum = DRUM_NOTE_MAP[next_note.pitch][0]

                if (curr_drum == next_drum and
                    0 < next_note.start - curr_note.start < window_ticks):
                    flams[i + 1].append(i)

        return flams

    def _detect_choke(self, notes: List[miditoolkit.Note]) -> Dict[int, int]:
        """
        チョークを検出 (Note Offが極端に短い)

        Returns:
            Dict[ノートインデックス, チョーク位置のtick]
        """
        chokes = {}
        for i, note in enumerate(notes):
            duration = note.end - note.start
            # シンバル類で極端に短い音価の場合、チョークとみなす
            if note.pitch in DRUM_NOTE_MAP:
                drum_type = DRUM_NOTE_MAP[note.pitch][0]
                if drum_type in ['CRASH', 'SPLASH', 'CHINA', 'RIDE_BOW', 'RIDE_BELL']:
                    if duration < DEFAULT_BEAT_RESOL / 4:  # 16分音符未満
                        chokes[i] = note.end

        return chokes

    def midi_to_tokens(self, midi_path: str) -> Tuple[List[str], List[int]]:
        """
        MIDIファイルをトークン列に変換

        Returns:
            tokens: トークン列
            bar_positions: 各小節の開始トークン位置
        """
        midi_obj = miditoolkit.MidiFile(midi_path)

        # ドラムトラックを探す
        drum_track = None
        for instrument in midi_obj.instruments:
            if instrument.is_drum:
                drum_track = instrument
                break

        if drum_track is None:
            raise ValueError("No drum track found in MIDI file")

        # 小節数を計算
        max_tick = max(note.end for note in drum_track.notes) if drum_track.notes else 0
        n_bars = int(np.ceil(max_tick / DEFAULT_BAR_RESOL))

        # フラムとチョークの検出
        flams = self._detect_flam(drum_track.notes)
        chokes = self._detect_choke(drum_track.notes)
        processed_notes = set()  # フラムの前打音をスキップするため

        tokens = []
        bar_positions = []

        for bar in range(n_bars):
            bar_positions.append(len(tokens))
            tokens.append('<BAR>')

            bar_start_tick = bar * DEFAULT_BAR_RESOL
            bar_end_tick = (bar + 1) * DEFAULT_BAR_RESOL

            # この小節内のすべてのイベント（ノートとチョーク）を収集
            events = []

            for note_idx, note in enumerate(drum_track.notes):
                if note_idx in processed_notes:
                    continue

                if bar_start_tick <= note.start < bar_end_tick:
                    tick_in_bar = note.start - bar_start_tick
                    beat = tick_in_bar // DEFAULT_BEAT_RESOL
                    pos_in_beat = (tick_in_bar % DEFAULT_BEAT_RESOL) * POSITIONS_PER_BEAT // DEFAULT_BEAT_RESOL

                    if note.pitch not in DRUM_NOTE_MAP:
                        continue

                    drum_type, technique = DRUM_NOTE_MAP[note.pitch]
                    velocity_level = self._velocity_to_level(note.velocity)

                    # フラムの処理
                    is_flam = note_idx in flams
                    if is_flam:
                        technique = 'FLAM'
                        # 前打音をスキップリストに追加
                        for grace_idx in flams[note_idx]:
                            processed_notes.add(grace_idx)

                    # ベロシティレベルが対応しているかチェック
                    if velocity_level == 'Ghost' and 'Ghost' not in self.vocab:
                        velocity_level = 'Normal'

                    # トークン生成
                    token = f'{drum_type}_{technique}_{velocity_level}'

                    # トークンが語彙に存在するか確認
                    if token not in self.token2idx:
                        # フォールバック: 基本的なHIT_Normalに戻す
                        token = f'{drum_type}_HIT_Normal'
                        if token not in self.token2idx:
                            continue

                    events.append((beat, pos_in_beat, tick_in_bar, token, 'note'))

                # チョークイベントの追加
                if note_idx in chokes:
                    choke_tick = chokes[note_idx]
                    if bar_start_tick <= choke_tick < bar_end_tick:
                        tick_in_bar = choke_tick - bar_start_tick
                        beat = tick_in_bar // DEFAULT_BEAT_RESOL
                        pos_in_beat = (tick_in_bar % DEFAULT_BEAT_RESOL) * POSITIONS_PER_BEAT // DEFAULT_BEAT_RESOL

                        drum_type = DRUM_NOTE_MAP[note.pitch][0]
                        choke_token = f'{drum_type}_CHOKE'

                        if choke_token in self.token2idx:
                            events.append((beat, pos_in_beat, tick_in_bar, choke_token, 'choke'))

            # イベントをソート（拍 > 位置 > 楽器の優先順位）
            events.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

            # トークン列に追加
            current_beat = -1
            current_pos = -1

            for beat, pos_in_beat, tick_in_bar, token, event_type in events:
                # 拍トークン
                if beat != current_beat:
                    tokens.append(f'<BEAT_{beat + 1}>')
                    current_beat = beat
                    current_pos = -1

                # 位置トークン
                if pos_in_beat != current_pos:
                    tokens.append(f'<POS_{pos_in_beat}>')
                    current_pos = pos_in_beat

                # 演奏トークン
                tokens.append(token)

        tokens.append('<EOS>')

        return tokens, bar_positions

    def tokens_to_indices(self, tokens: List[str]) -> np.ndarray:
        """トークン列をインデックス列に変換"""
        indices = []
        for token in tokens:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                # 未知のトークンはPADとして扱う
                indices.append(self.token2idx['<PAD>'])
        return np.array(indices, dtype=np.int32)

    def indices_to_tokens(self, indices: np.ndarray) -> List[str]:
        """インデックス列をトークン列に変換"""
        return [self.idx2token.get(idx, '<PAD>') for idx in indices]

    def save_vocab(self, vocab_path: str):
        """語彙を保存"""
        import pickle
        with open(vocab_path, 'wb') as f:
            pickle.dump((self.token2idx, self.idx2token), f)

    def load_vocab(self, vocab_path: str):
        """語彙を読み込み"""
        import pickle
        with open(vocab_path, 'rb') as f:
            self.token2idx, self.idx2token = pickle.load(f)
            self.vocab = list(self.token2idx.keys())
            self.vocab_size = len(self.vocab)


if __name__ == '__main__':
    # テスト用
    tokenizer = DrumTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"\nFirst 50 tokens:")
    for i, token in enumerate(tokenizer.vocab[:50]):
        print(f"  {i:3d}: {token}")

    print(f"\nLast 20 tokens:")
    for i, token in enumerate(tokenizer.vocab[-20:], start=len(tokenizer.vocab)-20):
        print(f"  {i:3d}: {token}")
