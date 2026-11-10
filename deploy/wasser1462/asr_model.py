#!/usr/bin/env python3
#
# Common ASR model interface: abstract base + implementation for the
# exported Qwen3-ASR ONNX pipeline. Builds directly on infer_qwen3_asr's
# helpers; no intermediate wrapper layer.

import os
import re
import sys
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import infer_qwen3_asr as _pipeline

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


class BaseASRModel(ABC):
    """ASR模型基类"""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None

    @abstractmethod
    def load(self):
        """加载模型"""
        pass

    @abstractmethod
    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        """转录一批音频"""
        pass


class Qwen3ASRModel(BaseASRModel):
    """Qwen3-ASR ONNX 实现。

    model_id: 导出 ONNX 文件所在目录（conv_frontend/encoder/decoder）。
    model_dir: Qwen3-ASR checkpoint 目录，提供 tokenizer 与 mel 特征。
    """

    def __init__(
        self,
        model_id: str,
        model_dir: str = "Qwen3-ASR-0.6B",
        device: str = "cpu",
        quant: str = "int8",
        chunk_size: int = 100,
        max_new_tokens: int = 64,
        max_total_len: int = 1024,
        context: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        super().__init__(model_id)
        self.model_dir = model_dir
        self.device = device
        self.quant = quant
        self.chunk_size = int(chunk_size)
        self.max_new_tokens = int(max_new_tokens)
        self.max_total_len = int(max_total_len)
        self.context = context
        self.language = language

    def load(self):
        _pipeline._check_model_dir(self.model_dir)
        if not _pipeline._register_qwen3_asr():
            raise RuntimeError(
                "failed to register the qwen3_asr processor; this pipeline needs "
                "transformers 4.x (check_model_inputs was removed in transformers 5)"
            )
        tok = _pipeline._load_tokenizer(self.model_dir)
        proc = _pipeline._load_processor(self.model_dir)
        _, audio_token_id = _pipeline._resolve_audio_token_and_id(tok, proc)

        conv = _pipeline._make_sess(os.path.join(self.model_id, "conv_frontend.onnx"), device=self.device)

        suffix = ".int8" if self.quant == "int8" else ""
        enc_path = os.path.join(self.model_id, f"encoder{suffix}.onnx")
        dec_path = os.path.join(self.model_id, f"decoder{suffix}.onnx")
        enc = _pipeline._make_sess_with_fallback(
            enc_path,
            device=self.device,
            fp32_fallback=enc_path.replace(".int8.onnx", ".onnx") if suffix else None,
        )
        dec = _pipeline._make_sess_with_fallback(
            dec_path,
            device=self.device,
            fp32_fallback=dec_path.replace(".int8.onnx", ".onnx") if suffix else None,
        )

        self.model = {
            "tok": tok,
            "proc": proc,
            "audio_token_id": audio_token_id,
            "conv": conv,
            "enc": enc,
            "dec": dec,
            "dec_out": [o.name for o in dec.get_outputs()],
        }
        return self

    def _encode(self, wav: np.ndarray):
        m = self.model
        text = _pipeline._build_text_prompt(m["proc"], self.context or "", self.language)
        combined = m["proc"](
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
        (conv_output,) = m["conv"].run(["conv_output"], {"input_features": mel_input})

        feat_len = (feat_mask != 0).sum(axis=1).astype(np.int64)
        a_len = _pipeline._feat_to_audio_tokens_len_np(feat_len, chunk_size=self.chunk_size)
        a_conv = int(conv_output.shape[1])
        pos = np.arange(a_conv, dtype=np.int64).reshape(1, a_conv)
        tok_mask = pos < a_len.reshape(-1, 1)

        (audio_features,) = m["enc"].run(
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
        slots = int((input_ids == m["audio_token_id"]).sum())
        if slots != a:
            raise RuntimeError(f"audio slot mismatch: prompt slots={slots}, encoder tokens={a}")
        return input_ids, prompt_attn, audio_features

    def _decode(self, input_ids: np.ndarray, prompt_attn: np.ndarray, audio_features: np.ndarray) -> List[int]:
        m = self.model
        n_layers, model_max_len, kv, hd = _pipeline._infer_cache_meta(m["dec"])
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
        out_names = m["dec_out"]

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
            out_map = dict(zip(out_names, m["dec"].run(out_names, feed)))
            for i in range(n_layers):
                caches[2 * i][:, cur_len : cur_len + s] = out_map[f"key_delta_{i}"]
                caches[2 * i + 1][:, cur_len : cur_len + s] = out_map[f"value_delta_{i}"]
            return np.asarray(out_map["logits"], dtype=np.float32)

        logits = run(input_ids, 0, prompt_attn)
        cur_len = s0
        eos_id = int(m["tok"].eos_token_id) if m["tok"].eos_token_id is not None else None

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

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        if isinstance(audio_batch, np.ndarray):
            raise TypeError("audio_batch must be a list of 1-D arrays; wrap a single array as [wav]")
        if self.model is None:
            raise RuntimeError("model not loaded; call load() first")

        results: List[str] = []
        for wav in audio_batch:
            wav = _to_mono_f32(wav)
            if wav.size == 0:
                raise ValueError("empty audio")
            input_ids, prompt_attn, audio_features = self._encode(wav)
            tokens = self._decode(input_ids, prompt_attn, audio_features)
            text = self.model["tok"].decode(tokens, skip_special_tokens=True)
            results.append(_LANG_PREFIX_RE.sub("", text).strip())
        return results
