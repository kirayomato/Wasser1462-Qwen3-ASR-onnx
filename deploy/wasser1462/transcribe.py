#!/usr/bin/env python3
#
# Thin inference wrapper around the exported Qwen3-ASR ONNX pipeline.
#
# Library API:
#     init(...)                      # configure / preload, optional
#     transcribe(wav: np.ndarray) -> str
#     batch_transcribe(audio_batch: List[np.ndarray]) -> List[str]
#
# CLI:
#     python transcribe.py --wav a.wav [--wav b.wav ...]

import argparse
import os
import re
import sys
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import infer_qwen3_asr as base  # noqa: E402

# Model output may start with "language {name}<asr_text>" (or the degraded
# "<asr text>" variant); strip it so transcribe() returns plain text.
_LANG_PREFIX_RE = re.compile(
    r"^language\s+\w+(?:<asr_text>|[\s\n]+asr\s+text\s*)", re.IGNORECASE
)


def _to_mono_f32(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    elif wav.dtype == np.int32:
        wav = wav.astype(np.float32) / 2147483648.0
    else:
        wav = wav.astype(np.float32, copy=False)
    return np.clip(wav, -1.0, 1.0)


class Transcriber:
    """Loads the ONNX pipeline once and transcribes 16 kHz mono waveforms."""

    def __init__(
        self,
        model_dir: str = "Qwen3-ASR-0.6B",
        onnx_dir: str = "model",
        device: str = "cpu",
        quant: str = "int8",
        chunk_size: int = 100,
        max_new_tokens: int = 64,
        max_total_len: int = 1024,
        context: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        base._check_model_dir(model_dir)
        base._register_qwen3_asr()
        self.tok = base._load_tokenizer(model_dir)
        self.proc = base._load_processor(model_dir)
        _, self.audio_token_id = base._resolve_audio_token_and_id(self.tok, self.proc)
        self.chunk_size = int(chunk_size)
        self.max_new_tokens = int(max_new_tokens)
        self.max_total_len = int(max_total_len)
        self.context = context
        self.language = language

        self.conv = base._make_sess(os.path.join(onnx_dir, "conv_frontend.onnx"), device=device)

        suffix = ".int8" if quant == "int8" else ""
        enc_path = os.path.join(onnx_dir, f"encoder{suffix}.onnx")
        dec_path = os.path.join(onnx_dir, f"decoder{suffix}.onnx")
        enc_fp32 = enc_path.replace(".int8.onnx", ".onnx") if suffix else None
        dec_fp32 = dec_path.replace(".int8.onnx", ".onnx") if suffix else None
        self.enc = base._make_sess_with_fallback(enc_path, device=device, fp32_fallback=enc_fp32)
        self.dec = base._make_sess_with_fallback(dec_path, device=device, fp32_fallback=dec_fp32)
        self._dec_out_names = [o.name for o in self.dec.get_outputs()]

    def _encode(self, wav: np.ndarray):
        text = base._build_text_prompt(self.proc, self.context or "", self.language)
        combined = self.proc(
            text=[text],
            audio=[wav],
            sampling_rate=16000,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        input_features = np.asarray(combined["input_features"], dtype=np.float32)
        feat_mask = np.asarray(combined["feature_attention_mask"], dtype=np.int32)
        input_ids = np.asarray(combined["input_ids"], dtype=np.int64)
        prompt_attn = np.asarray(combined["attention_mask"], dtype=np.int64)

        mel_input = input_features.transpose(0, 2, 1)
        (conv_output,) = self.conv.run(["conv_output"], {"input_features": mel_input})

        feat_len = (feat_mask != 0).sum(axis=1).astype(np.int64)
        a_len = base._feat_to_audio_tokens_len_np(feat_len, chunk_size=self.chunk_size)
        a_conv = int(conv_output.shape[1])
        pos = np.arange(a_conv, dtype=np.int64).reshape(1, a_conv)
        tok_mask = pos < a_len.reshape(-1, 1)

        (audio_features,) = self.enc.run(
            ["audio_features"],
            {
                "input_features": conv_output,
                "feature_attention_mask": tok_mask.astype(np.bool_),
            },
        )
        audio_features = np.asarray(audio_features, dtype=np.float32)
        a = int(a_len[0])
        if a <= 0 or a > int(audio_features.shape[1]):
            raise RuntimeError(
                f"invalid audio token length: {a} (encoder out {audio_features.shape[1]})"
            )
        audio_features = audio_features[:, :a, :]
        slots = int((input_ids == self.audio_token_id).sum())
        if slots != a:
            raise RuntimeError(f"audio slot mismatch: prompt slots={slots}, encoder tokens={a}")
        return input_ids, prompt_attn, audio_features

    def _decode(self, input_ids: np.ndarray, prompt_attn: np.ndarray, audio_features: np.ndarray) -> List[int]:
        n_layers, model_max_len, kv, hd = base._infer_cache_meta(self.dec)
        if kv is None or hd is None:
            raise RuntimeError("decoder cache shape has dynamic kv/hd")
        s0 = int(input_ids.shape[1])
        total_len = int(model_max_len) if model_max_len is not None else self.max_total_len
        if s0 + self.max_new_tokens > total_len:
            raise RuntimeError(
                f"max_total_len not enough: S0({s0}) + max_new_tokens({self.max_new_tokens}) "
                f"> total_len({total_len})"
            )
        caches: List[np.ndarray] = [
            np.zeros((1, total_len, kv, hd), dtype=np.float32) for _ in range(2 * n_layers)
        ]

        def run(step_ids: np.ndarray, cur_len: int, attn: Optional[np.ndarray]) -> np.ndarray:
            s = int(step_ids.shape[1])
            if attn is None:
                attn = np.ones((1, s), dtype=np.int64)
            feed = {
                "input_ids": step_ids,
                "audio_features": audio_features,
                "attention_mask": attn,
                "cache_position": np.arange(cur_len, cur_len + s, dtype=np.int64),
            }
            for i in range(n_layers):
                feed[f"cache_key_{i}"] = caches[2 * i]
                feed[f"cache_value_{i}"] = caches[2 * i + 1]
            out_map = dict(zip(self._dec_out_names, self.dec.run(self._dec_out_names, feed)))
            for i in range(n_layers):
                caches[2 * i][:, cur_len : cur_len + s] = out_map[f"key_delta_{i}"]
                caches[2 * i + 1][:, cur_len : cur_len + s] = out_map[f"value_delta_{i}"]
            return np.asarray(out_map["logits"], dtype=np.float32)

        logits = run(input_ids, 0, prompt_attn)
        cur_len = s0
        eos_id = int(self.tok.eos_token_id) if self.tok.eos_token_id is not None else None

        out: List[int] = [int(np.argmax(logits[0, -1, :]))]
        for _ in range(self.max_new_tokens - 1):
            if eos_id is not None and out[-1] == eos_id:
                break
            logits = run(np.asarray([[out[-1]]], dtype=np.int64), cur_len, None)
            cur_len += 1
            out.append(int(np.argmax(logits[0, -1, :])))
        while out and eos_id is not None and out[-1] == eos_id:
            out.pop()
        return out

    def transcribe(self, wav: np.ndarray) -> str:
        wav = _to_mono_f32(wav)
        if wav.size == 0:
            raise ValueError("empty audio")
        input_ids, prompt_attn, audio_features = self._encode(wav)
        tokens = self._decode(input_ids, prompt_attn, audio_features)
        text = self.tok.decode(tokens, skip_special_tokens=True)
        return _LANG_PREFIX_RE.sub("", text).strip()

    def batch_transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        return [self.transcribe(w) for w in audio_batch]


# ---------------------------------------------------------------------------
# Module-level convenience API (lazy default instance)
# ---------------------------------------------------------------------------

_default: Optional[Transcriber] = None


def init(**kwargs) -> Transcriber:
    """Configure the module-level transcriber (see Transcriber for options)."""
    global _default
    _default = Transcriber(**kwargs)
    return _default


def _get() -> Transcriber:
    global _default
    if _default is None:
        _default = Transcriber()
    return _default


def transcribe(wav: np.ndarray) -> str:
    return _get().transcribe(wav)


def batch_transcribe(audio_batch: List[np.ndarray]) -> List[str]:
    return _get().batch_transcribe(audio_batch)


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-ASR ONNX transcription")
    p.add_argument("--wav", nargs="+", required=True, help=".wav or .npy files")
    p.add_argument("--model", default="Qwen3-ASR-0.6B", help="Qwen3-ASR checkpoint dir")
    p.add_argument("--onnx-dir", default="model", help="Dir with exported ONNX files")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--quant", choices=["fp32", "int8"], default="int8")
    p.add_argument("--chunk-size", type=int, default=100, help="Must match export")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-total-len", type=int, default=1024)
    p.add_argument("--context", default=None, help="System-message context (hotwords)")
    p.add_argument("--language", default=None, help="Forced language, e.g. English")
    args = p.parse_args()

    tr = Transcriber(
        model_dir=args.model,
        onnx_dir=args.onnx_dir,
        device=args.device,
        quant=args.quant,
        chunk_size=args.chunk_size,
        max_new_tokens=args.max_new_tokens,
        max_total_len=args.max_total_len,
        context=args.context,
        language=args.language,
    )
    for path in args.wav:
        print(tr.transcribe(base._load_audio_any(path)))


if __name__ == "__main__":
    main()
