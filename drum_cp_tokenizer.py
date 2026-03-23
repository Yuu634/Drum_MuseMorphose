"""
四肢ベースCPトークナイザー

CPイベント:
  [Position, Hand1, Hand2, Right_Foot, Left_Foot]

非CPイベント:
  <TEMPO_xxx>, <BAR>, <BEAT_x>, <EOS>
"""
import pickle
from typing import Dict, List, Tuple

import miditoolkit
import numpy as np

from drum_tokenizer import (
    DEFAULT_BAR_RESOL,
    DEFAULT_BEAT_RESOL,
    DRUM_NOTE_MAP,
    POSITIONS_PER_BEAT,
    DrumTokenizer,
)


class DrumCPTokenizer:
    def __init__(self):
        self.base = DrumTokenizer()

        self.event_type2idx = {
            "STRUCT": 0,
            "CP": 1,
            "<PAD_EVENT>": 2,
        }
        self.idx2event_type = {v: k for k, v in self.event_type2idx.items()}

        self.struct_vocab = self._build_struct_vocab()
        self.struct_token2idx = {tok: i for i, tok in enumerate(self.struct_vocab)}
        self.idx2struct_token = {i: tok for tok, i in self.struct_token2idx.items()}

        self.limb_vocab = self._build_limb_vocab()
        self.limb_token2idx = {tok: i for i, tok in enumerate(self.limb_vocab)}
        self.idx2limb_token = {i: tok for tok, i in self.limb_token2idx.items()}

        # 既存コード互換: prepare_drum_dataset.py が tokenizer.vocab_size を参照する
        self.struct_vocab_size = len(self.struct_vocab)
        self.limb_vocab_size = len(self.limb_vocab)
        self.event_type_vocab_size = len(self.event_type2idx)
        self.vocab_size = self.struct_vocab_size

        # 0-23 を有効POS, 24 を PAD_POS として使用
        self.pos_pad_value = 24

    def _build_struct_vocab(self) -> List[str]:
        vocab = ["<PAD>", "<EOS>", "<BAR>"]
        for tempo in range(60, 245, 5):
            vocab.append(f"<TEMPO_{tempo}>")
        vocab.extend([f"<BEAT_{i}>" for i in range(1, 5)])
        return vocab

    def _build_limb_vocab(self) -> List[str]:
        perf_tokens = [tok for tok in self.base.vocab if not tok.startswith("<")]
        return ["<PAD>", "<NONE>"] + sorted(perf_tokens)

    def _is_hand_note(self, drum_type: str) -> bool:
        return drum_type not in ("KICK", "HH_PEDAL")

    def _assign_feet(self, drum_type: str, token: str, rf: str, lf: str) -> Tuple[str, str]:
        if drum_type == "HH_PEDAL":
            if lf == "<NONE>":
                return rf, token
            return rf, lf

        if drum_type == "KICK":
            if rf == "<NONE>":
                return token, lf
            if lf == "<NONE>":
                return rf, token
            return rf, lf

        return rf, lf

    def _hand_priority(self, token: str) -> Tuple[int, int, str]:
        # 安定ソートのため、楽器優先とベロシティ優先を組み合わせる
        if token.startswith("SNARE"):
            inst_rank = 0
        elif token.startswith("HH"):
            inst_rank = 1
        elif token.startswith("RIDE"):
            inst_rank = 2
        elif token.startswith("CRASH"):
            inst_rank = 3
        elif token.startswith("TOM") or token.startswith("FLOOR"):
            inst_rank = 4
        else:
            inst_rank = 5

        if token.endswith("_Accent"):
            vel_rank = 0
        elif token.endswith("_Normal"):
            vel_rank = 1
        else:
            vel_rank = 2

        return (inst_rank, vel_rank, token)

    def _parse_note_to_token(self, note: miditoolkit.Note) -> Tuple[str, str]:
        drum_type, technique = DRUM_NOTE_MAP[note.pitch]
        velocity_level = self.base._velocity_to_level(note.velocity)
        token = f"{drum_type}_{technique}_{velocity_level}"

        if token not in self.base.token2idx:
            fallback = f"{drum_type}_HIT_Normal"
            if fallback in self.base.token2idx:
                token = fallback
            else:
                return drum_type, ""
        return drum_type, token

    def _quantize_tempo(self, bpm: float) -> int:
        return self.base._quantize_tempo(bpm)

    def _get_tempo_at_tick(self, midi_obj: miditoolkit.MidiFile, tick: int) -> float:
        return self.base._get_tempo_at_tick(midi_obj, tick)

    def _infer_effective_ticks_per_beat(
        self, midi_obj: miditoolkit.MidiFile, notes: List[miditoolkit.Note]
    ) -> int:
        return self.base._infer_effective_ticks_per_beat(midi_obj, notes)

    def save_vocab(self, vocab_path: str):
        payload = {
            "format": "cp_limb_v1",
            "tokenization_method": "cp_limb_v1",
            "event_type2idx": self.event_type2idx,
            "idx2event_type": self.idx2event_type,
            "struct_token2idx": self.struct_token2idx,
            "idx2struct_token": self.idx2struct_token,
            "limb_token2idx": self.limb_token2idx,
            "idx2limb_token": self.idx2limb_token,
            "pos_pad_value": self.pos_pad_value,
            "positions_per_beat": POSITIONS_PER_BEAT,
        }
        with open(vocab_path, "wb") as f:
            pickle.dump(payload, f)

    def midi_to_cp_data(self, midi_path: str) -> Tuple[Dict[str, List[int]], List[int]]:
        midi_obj = miditoolkit.MidiFile(midi_path)

        drum_track = None
        for instrument in midi_obj.instruments:
            if instrument.is_drum:
                drum_track = instrument
                break

        if drum_track is None:
            raise ValueError("No drum track found in MIDI file")

        effective_tpb = self._infer_effective_ticks_per_beat(midi_obj, drum_track.notes)
        scale_factor = DEFAULT_BEAT_RESOL / effective_tpb
        if scale_factor != 1.0:
            for note in drum_track.notes:
                note.start = int(note.start * scale_factor)
                note.end = int(note.end * scale_factor)
            for tempo_change in midi_obj.tempo_changes:
                tempo_change.time = int(tempo_change.time * scale_factor)

        max_tick = max(note.end for note in drum_track.notes) if drum_track.notes else 0
        n_bars = int(np.ceil(max_tick / DEFAULT_BAR_RESOL))

        event_type_seq = []
        struct_token_seq = []
        cp_pos_seq = []
        cp_hand1_seq = []
        cp_hand2_seq = []
        cp_rf_seq = []
        cp_lf_seq = []
        bar_positions = []

        struct_pad = self.struct_token2idx["<PAD>"]
        limb_pad = self.limb_token2idx["<PAD>"]

        def append_struct(struct_token: str):
            event_type_seq.append(self.event_type2idx["STRUCT"])
            struct_token_seq.append(self.struct_token2idx[struct_token])
            cp_pos_seq.append(self.pos_pad_value)
            cp_hand1_seq.append(limb_pad)
            cp_hand2_seq.append(limb_pad)
            cp_rf_seq.append(limb_pad)
            cp_lf_seq.append(limb_pad)

        def append_cp(pos: int, hand1: str, hand2: str, rf: str, lf: str):
            event_type_seq.append(self.event_type2idx["CP"])
            struct_token_seq.append(struct_pad)
            cp_pos_seq.append(pos)
            cp_hand1_seq.append(self.limb_token2idx[hand1])
            cp_hand2_seq.append(self.limb_token2idx[hand2])
            cp_rf_seq.append(self.limb_token2idx[rf])
            cp_lf_seq.append(self.limb_token2idx[lf])

        for bar in range(n_bars):
            bar_positions.append(len(event_type_seq))

            bar_start_tick = bar * DEFAULT_BAR_RESOL
            bar_end_tick = (bar + 1) * DEFAULT_BAR_RESOL

            quantized_tempo = self._quantize_tempo(self._get_tempo_at_tick(midi_obj, bar_start_tick))
            append_struct(f"<TEMPO_{quantized_tempo}>")
            append_struct("<BAR>")

            tick_events: Dict[int, List[Tuple[str, str]]] = {}
            for note in drum_track.notes:
                if not (bar_start_tick <= note.start < bar_end_tick):
                    continue
                if note.pitch not in DRUM_NOTE_MAP:
                    continue

                drum_type, token = self._parse_note_to_token(note)
                if not token:
                    continue

                tick_in_bar = note.start - bar_start_tick
                tick_events.setdefault(tick_in_bar, []).append((drum_type, token))

            current_beat = -1
            for tick_in_bar in sorted(tick_events.keys()):
                beat = tick_in_bar // DEFAULT_BEAT_RESOL
                pos_in_beat = (tick_in_bar % DEFAULT_BEAT_RESOL) * POSITIONS_PER_BEAT // DEFAULT_BEAT_RESOL

                if beat != current_beat:
                    append_struct(f"<BEAT_{beat + 1}>")
                    current_beat = beat

                hand_candidates = []
                rf = "<NONE>"
                lf = "<NONE>"

                for drum_type, token in tick_events[tick_in_bar]:
                    if self._is_hand_note(drum_type):
                        hand_candidates.append(token)
                    else:
                        rf, lf = self._assign_feet(drum_type, token, rf, lf)

                hand_candidates = sorted(hand_candidates, key=self._hand_priority)
                hand1 = hand_candidates[0] if len(hand_candidates) > 0 else "<NONE>"
                hand2 = hand_candidates[1] if len(hand_candidates) > 1 else "<NONE>"

                append_cp(pos_in_beat, hand1, hand2, rf, lf)

        append_struct("<EOS>")

        data = {
            "event_type": event_type_seq,
            "struct_token": struct_token_seq,
            "cp_pos": cp_pos_seq,
            "cp_hand1": cp_hand1_seq,
            "cp_hand2": cp_hand2_seq,
            "cp_right_foot": cp_rf_seq,
            "cp_left_foot": cp_lf_seq,
        }
        return data, bar_positions
