#!/usr/bin/env python3
# Copyright    2025                            (authors: Yifan Yang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Usage:

./bin/evalute.py \
    --input-dir /path/to/samples \
    --metrics "asr, asv" \
    --asr-model "hubert" \
    --asv-model "wavlm_ecapa_tdnn"

"""
import argparse
import logging
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Path to audio samples.",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="asr, asv",
        help="Comma-separated list of evaluation metrics (asr, asv).",
    )

    parser.add_argument(
        "--asr-model",
        type=str,
        choices=["hubert", "whisper"],
        default="hubert",
        help="Choose which ASR model to use.",
    )

    parser.add_argument(
        "--asv-model",
        type=str,
        choices=["wavlm_ecapa_tdnn"],
        default="wavlm_ecapa_tdnn",
        help="Choose which ASV model to use.",
    )

    return parser.parse_args()


def get_asr_model(model_name, device):
    if model_name == "hubert":
        from speech_recognition.hubert import HubertASRModel

        return HubertASRModel(device=device)

    elif model_name == "whisper":
        from speech_recognition.whisper import WhisperASRModel

        return WhisperASRModel(device=device)

    else:
        raise ValueError(f"Unsupported ASR model: {model_name}")


def get_asv_model(model_name, device):
    if model_name == "wavlm_ecapa_tdnn":
        from speaker_verification.wavlm_ecapa_tdnn import WavlmECAPAASVModel

        return WavlmECAPAASVModel(device=device)

    else:
        raise ValueError(f"Unsupported ASV model: {model_name}")


@torch.no_grad()
def main(args):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    # Setup ASR model
    asr_model = None
    if "asr" in args.metrics:
        asr_model = get_asr_model(args.asr_model, device)

    # Setup ASV model
    asv_model = None
    if "asv" in args.metrics:
        asv_model = get_asv_model(args.asv_model, device)

    results = defaultdict(dict)
    audio_files = sorted(
        [
            f
            for f in args.input_dir.iterdir()
            if f.is_file() and f.name.endswith(".wav") and f.name[:-4].isdigit()
        ],
        key=lambda f: int(f.name[:-4]),
    )

    for audio_file in audio_files:
        audio_id = audio_file.stem
        audio, sr = torchaudio.load(audio_file)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)

        prompt_audio_file = (
            audio_file.parent / f"{audio_file.stem}-prompt{audio_file.suffix}"
        )
        prompt_audio, sr = torchaudio.load(prompt_audio_file)
        prompt_audio = torchaudio.transforms.Resample(sr, 16000)(prompt_audio)

        text_file = audio_file.parent / f"{audio_file.stem}.txt"
        with open(text_file) as f:
            text = f.read().strip()

        if "asr" in args.metrics:
            hyp = asr_model.transcribe(audio)
            hyp = asr_model.text_normalize(hyp)
            ref = asr_model.text_normalize(text)

            error, total, wer = asr_model.compute_wer(hyp, ref).values()

            results[audio_id]["error"] = error
            results[audio_id]["total"] = total

            logging.info(f"hyp: {hyp}, ref: {ref}, WER: {wer}")

        if "asv" in args.metrics:
            sim = asv_model.compute_similarity(prompt_audio, audio)
            results[audio_id]["sim"] = sim
            logging.info(f"SIM: {sim}")

    if "asr" in args.metrics:
        overall_wer = sum(sample["error"] for sample in results.values()) / sum(
            sample["total"] for sample in results.values()
        )
        logging.info(f"Overall WER: {overall_wer}")

    if "asv" in args.metrics:
        overall_sim = sum(sample["sim"] for sample in results.values()) / len(
            results.values()
        )
        logging.info(f"Overall SIM: {overall_sim}")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    args = get_args()
    args.metrics = args.metrics.replace(" ", "").split(",")
    logging.basicConfig(
        filename=f"{args.input_dir.name}.log",
        level=logging.INFO,
        format=formatter,
    )
    logging.info(vars(args))
    main(args)
