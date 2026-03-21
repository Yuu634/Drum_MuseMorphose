"""
ドラムトークン列からMIDIファイルへの変換
"""
import numpy as np
import miditoolkit
from typing import List, Optional

# 定数
DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
POSITIONS_PER_BEAT = 24
DEFAULT_BPM = 120

# ドラム楽器からMIDIノート番号へのマッピング（逆引き）
# drum_tokenizer.pyのDRUM_NOTE_MAPと対応させる
DRUM_TO_NOTE = {
    'KICK': 36,
    'SNARE': 38,           # D2 (MIDI 38)
    'SNARE_ACCENT': 37,    # C#2 (MIDI 37) - Snare accent
    'SNARE_XSTICK': 39,
    'SNARE_RIMSHOT': 40,
    'TOM1': 48,            # C3 (MIDI 48)
    'TOM2': 47,            # B2 (MIDI 47)
    'FLOOR': 43,           # G2 (MIDI 43)
    'HH_CLOSED': 42,       # F#2 (MIDI 42)
    'HH_HALFOPEN': 44,     # G#2
    'HH_OPEN': 46,         # A#2 (MIDI 46)
    'HH_PEDAL': 44,        # G#2 (MIDI 44)
    'RIDE_BOW': 51,        # D#3 (MIDI 51)
    'RIDE_BELL': 53,
    'CRASH': 49,
    'SPLASH': 55,
    'CHINA': 52,           # E3 (MIDI 52)
    'TAMBOURINE': 54,      # F#3 (MIDI 54)
    'COWBELL': 56,         # G#3 (MIDI 56)
}

# ベロシティレベルのマッピング
VELOCITY_MAP = {
    'Ghost': 30,
    'Normal': 80,
    'Accent': 120,
}


class DrumToken2MIDI:
    """ドラムトークン列をMIDIに変換"""

    def __init__(self, default_bpm: int = 120):
        self.default_bpm = default_bpm

    def _parse_drum_token(self, token: str) -> Optional[tuple]:
        """
        ドラムトークンをパース

        Returns:
            (instrument, technique, velocity_level) or None
        """
        if token.startswith('<') or token in ['<PAD>', '<EOS>']:
            return None

        parts = token.split('_')
        if len(parts) < 3:
            return None

        # 楽器名を構築（2単語の可能性がある: HH_CLOSED, RIDE_BOWなど）
        if parts[1] in ['CLOSED', 'HALFOPEN', 'OPEN', 'PEDAL', 'BOW', 'BELL', 'ACCENT']:
            instrument = f"{parts[0]}_{parts[1]}"
            technique = parts[2]
            velocity_level = '_'.join(parts[3:]) if len(parts) > 3 else 'Normal'
        else:
            instrument = parts[0]
            technique = parts[1]
            velocity_level = '_'.join(parts[2:]) if len(parts) > 2 else 'Normal'

        return instrument, technique, velocity_level

    def _get_velocity(self, velocity_level: str) -> int:
        """ベロシティレベルをMIDIベロシティ値に変換"""
        return VELOCITY_MAP.get(velocity_level, VELOCITY_MAP['Normal'])

    def _get_note_number(self, instrument: str, technique: str) -> Optional[int]:
        """楽器と技法からMIDIノート番号を取得"""
        if technique == 'XSTICK':
            return DRUM_TO_NOTE.get(f'{instrument}_XSTICK')
        elif technique == 'RIMSHOT':
            return DRUM_TO_NOTE.get(f'{instrument}_RIMSHOT')
        elif technique == 'ACCENT':
            return DRUM_TO_NOTE.get(f'{instrument}_ACCENT')
        else:
            return DRUM_TO_NOTE.get(instrument)

    def _calculate_duration(self, technique: str) -> int:
        """技法に応じた音価を計算（ticks）"""
        if technique == 'ROLL':
            return DEFAULT_BEAT_RESOL * 2  # 2拍分
        elif technique == 'FLAM':
            return DEFAULT_BEAT_RESOL // 4  # 16分音符
        else:
            return DEFAULT_BEAT_RESOL // 8  # 32分音符（デフォルト）

    def tokens_to_midi(
        self,
        tokens: List[str],
        output_path: Optional[str] = None,
        bpm: Optional[int] = None
    ) -> miditoolkit.MidiFile:
        """
        トークン列をMIDIファイルに変換

        Args:
            tokens: トークン列
            output_path: 出力MIDIファイルパス（Noneの場合は保存しない）
            bpm: テンポ（BPM）

        Returns:
            MidiFile オブジェクト
        """
        if bpm is None:
            bpm = self.default_bpm

        # MIDIオブジェクト作成
        midi_obj = miditoolkit.MidiFile()
        midi_obj.ticks_per_beat = DEFAULT_BEAT_RESOL

        # ドラムトラック作成
        drum_track = miditoolkit.Instrument(program=0, is_drum=True, name='Drums')

        # テンポ設定
        midi_obj.tempo_changes.append(
            miditoolkit.TempoChange(bpm, 0)
        )

        # トークン列から最大小節番号を取得（<BAR>トークンの総数 = 小節総数）
        max_bar = 0
        for token in tokens:
            if token == '<BAR>':
                max_bar += 1
        
        # 予期される全小節数を記録（空の小節を含む）
        total_bars = max_bar

        # トークンを解析してノートイベントを生成
        current_bar = 0
        current_beat = 0
        current_pos = 0
        current_tick = 0

        choke_events = []  # チョークイベントのリスト

        for i, token in enumerate(tokens):
            if token == '<BAR>':
                # 新しい小節を開始（空の小節も含める）
                current_bar += 1
                current_beat = 0
                current_pos = 0
                current_tick = (current_bar - 1) * DEFAULT_BAR_RESOL
                
                # 空小節追加
                if tokens[i+1] == '<BAR>':
                    midi_obj.markers.append(
                        miditoolkit.Marker(f'Bar-{current_bar}', (current_bar - 1) * DEFAULT_BAR_RESOL)
                    )

            elif token.startswith('<BEAT_'):
                beat_num = int(token.replace('<BEAT_', '').replace('>', ''))
                current_beat = beat_num - 1
                current_pos = 0
                current_tick = (current_bar - 1) * DEFAULT_BAR_RESOL + current_beat * DEFAULT_BEAT_RESOL

            elif token.startswith('<POS_'):
                pos_num = int(token.replace('<POS_', '').replace('>', ''))
                current_pos = pos_num
                tick_in_beat = (DEFAULT_BEAT_RESOL * current_pos) // POSITIONS_PER_BEAT
                current_tick = (current_bar - 1) * DEFAULT_BAR_RESOL + current_beat * DEFAULT_BEAT_RESOL + tick_in_beat

            elif token == '<EOS>':
                break

            elif token not in ['<PAD>']:
                # ドラム演奏トークンの処理
                parsed = self._parse_drum_token(token)
                if parsed is None:
                    continue

                instrument, technique, velocity_level = parsed

                # CHOKEトークンの処理
                if technique == 'CHOKE':
                    # 最後に追加されたこの楽器のノートを探してチョークを適用
                    for note in reversed(drum_track.notes):
                        note_instrument = None
                        for key, val in DRUM_TO_NOTE.items():
                            if val == note.pitch:
                                note_instrument = key.split('_')[0]
                                break

                        if note_instrument == instrument.split('_')[0]:
                            # このノートをチョーク（音を短くする）
                            note.end = min(note.end, current_tick)
                            break
                    continue

                # MIDIノート番号を取得
                note_number = self._get_note_number(instrument, technique)
                if note_number is None:
                    continue

                # ベロシティを取得
                velocity = self._get_velocity(velocity_level)

                # 音価を計算
                duration = self._calculate_duration(technique)

                # FLAMの処理（前打音 + 主音符）
                if technique == 'FLAM':
                    # 前打音（少し早めにゴーストノート）
                    grace_note = miditoolkit.Note(
                        velocity=VELOCITY_MAP['Ghost'],
                        pitch=note_number,
                        start=max(0, current_tick - 30),
                        end=max(0, current_tick - 10)
                    )
                    drum_track.notes.append(grace_note)

                    # 主音符
                    main_note = miditoolkit.Note(
                        velocity=velocity,
                        pitch=note_number,
                        start=current_tick,
                        end=current_tick + duration
                    )
                    drum_track.notes.append(main_note)

                elif technique == 'ROLL':
                    # ロールの表現（複数の短いノートで表現）
                    roll_duration = DEFAULT_BEAT_RESOL * 2
                    roll_subdivision = DEFAULT_BEAT_RESOL // 16  # 64分音符で刻む

                    for t in range(0, roll_duration, roll_subdivision):
                        note = miditoolkit.Note(
                            velocity=max(40, velocity - 20),  # 少し弱めに
                            pitch=note_number,
                            start=current_tick + t,
                            end=current_tick + t + roll_subdivision // 2
                        )
                        drum_track.notes.append(note)

                else:
                    # 通常のノート
                    note = miditoolkit.Note(
                        velocity=velocity,
                        pitch=note_number,
                        start=current_tick,
                        end=current_tick + duration
                    )
                    drum_track.notes.append(note)

        # ドラムトラックを追加
        midi_obj.instruments.append(drum_track)

        # 全小節のマーカーを追加（空の小節を含む）
        # これにより、元譜面の全小節が再構築される
        #for bar in range(total_bars):
        #    midi_obj.markers.append(
        #        miditoolkit.Marker(f'Bar-{bar + 1}', bar * DEFAULT_BAR_RESOL)
        #    )

        # MIDIの終了時刻を設定（全小節を包含するように）
        midi_obj.end_time = total_bars * DEFAULT_BAR_RESOL

        # ファイル保存
        if output_path is not None:
            midi_obj.dump(output_path)

        return midi_obj


def tokens_to_midi(
    tokens: List[str],
    output_path: str,
    bpm: int = 120
) -> miditoolkit.MidiFile:
    """
    ユーティリティ関数: トークン列をMIDIファイルに変換

    Args:
        tokens: トークン列
        output_path: 出力MIDIファイルパス
        bpm: テンポ（BPM）

    Returns:
        MidiFile オブジェクト
    """
    converter = DrumToken2MIDI(default_bpm=bpm)
    return converter.tokens_to_midi(tokens, output_path, bpm)


if __name__ == '__main__':
    # テスト用
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

    print("Converting test tokens to MIDI...")
    midi_obj = tokens_to_midi(test_tokens, '/tmp/test_drum_output.mid', bpm=120)
    print(f"Created MIDI with {len(midi_obj.instruments[0].notes)} notes")
    print("Saved to: /tmp/test_drum_output.mid")
