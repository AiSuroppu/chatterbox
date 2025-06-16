from dataclasses import dataclass, replace
from pathlib import Path
from typing import Union, List, Optional

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
        compile_mode: str = None
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        if compile_mode:
            self.t3 = torch.compile(self.t3, mode=compile_mode)
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device, compile_mode: str = None) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds, compile_mode=compile_mode)

    @classmethod
    def from_pretrained(cls, device, compile_mode: str = None) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device, compile_mode=compile_mode)

    def create_voice_embedding(self, wav_fpath: str, exaggeration: float = 0.5) -> Conditionals:
        """
        Creates a voice embedding object from a given audio file. This object can be
        cached and reused in subsequent calls to `generate` for efficiency.

        Args:
            wav_fpath (str): Path to the audio file to use as a voice reference.
            exaggeration (float, optional): Controls the emotional exaggeration of the voice. Defaults to 0.5.

        Returns:
            Conditionals: An object containing the necessary conditioning information for TTS.
        """
        # Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)

        return Conditionals(t3_cond, s3gen_ref_dict)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        """
        [DEPRECATED] Prepares and sets an internal voice embedding from an audio file.
        Prefer using `create_voice_embedding` and passing the result to `generate`
        via the `voice_embedding_cache` argument for better performance.
        This method is maintained for backward compatibility.
        """
        self.conds = self.create_voice_embedding(wav_fpath, exaggeration)

    def generate(
        self,
        text: Union[str, List[str]],
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        use_analyzer=False,
        skip_error_check=False,
        voice_embedding_cache=None,
        offload_s3gen=False,
        offload_t3=False,
        batch_size=1,
    ):
        """
        Generates audio from text.

        Args:
            text (Union[str, List[str]]): The text or a batch of texts to be synthesized.
            repetition_penalty (float): Penalty for repeating tokens.
            min_p (float): Minimum probability for nucleus sampling.
            top_p (float): Cumulative probability for nucleus sampling.
            audio_prompt_path (str, optional): Path to an audio file to use as a voice reference.
            exaggeration (float): Controls the intensity of the emotion.
            cfg_weight (float): Classifier-Free Guidance weight.
            temperature (float): Sampling temperature.
            use_analyzer (bool): Whether to use the alignment stream analyzer.
            skip_error_check (bool): If True and use_analyzer is True, ignores errors from the
                alignment stream analyzer and attempts to generate audio anyway.
            voice_embedding_cache (Conditionals, optional): A pre-computed voice embedding object.
                Using this is more efficient than `audio_prompt_path` for multiple generations with the same voice.
            offload_s3gen (bool): If True, offloads the s3gen model to the CPU
                during T3 inference to save VRAM, allowing for larger batch sizes.
                This adds a small time overhead for model transfer.
            offload_t3 (bool): If True, offloads the T3 model to CPU after its inference
                to free up VRAM for s3gen.
            batch_size (int): The number of texts to process in a single batch.
        """
        # Determine which conditionals to use, ensuring valid arguments.
        if voice_embedding_cache is not None and audio_prompt_path is not None:
            raise ValueError("Cannot provide both `voice_embedding_cache` and `audio_prompt_path`. Please use one.")

        if voice_embedding_cache is not None:
            active_conds = voice_embedding_cache
        elif audio_prompt_path is not None:
            # For backward compatibility, create and set the internal cache.
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            active_conds = self.conds
        else:
            # Use the internal cache if it exists (e.g., from a built-in voice or previous call).
            active_conds = self.conds

        if active_conds is None:
            raise ValueError("No voice conditioning provided. Please set `voice_embedding_cache` or `audio_prompt_path`.")
        active_conds = active_conds.to(self.device)

        # Handle exaggeration: If the user provides a different exaggeration value,
        # create a new T3Cond for this generation call without modifying the original cache object.
        t3_cond_for_inference = active_conds.t3
        if exaggeration != t3_cond_for_inference.emotion_adv[0, 0, 0]:
            t3_cond_for_inference = replace(
                t3_cond_for_inference,
                emotion_adv=exaggeration * torch.ones(1, 1, 1)
            )

        if isinstance(text, str):
            is_list_input = False
            texts = [text]
        else:
            is_list_input = True
            texts = text

        # Pre-process, tokenize, and sort by length for efficient batching
        tokenized_items = []
        for i, t in enumerate(texts):
            norm_t = punc_norm(t)
            tokens = self.tokenizer.text_to_tokens(norm_t).squeeze(0)
            tokenized_items.append({'original_index': i, 'tokens': tokens, 'length': len(tokens)})

        tokenized_items.sort(key=lambda x: x['length'], reverse=True)

        # Offload s3gen to CPU once before T3 inference loop
        if offload_s3gen and self.device.type != 'cpu':
            self.s3gen.to('cpu')
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Ensure T3 is on the correct device for inference.
        # This is safe to do here because s3gen may have just been offloaded.
        if self.t3.device != self.device:
            self.t3.to(self.device)

        # T3 inference in mini-batches
        all_speech_tokens_sorted = []
        all_error_flags_sorted = []
        with torch.inference_mode():
            for i in range(0, len(tokenized_items), batch_size):
                mini_batch_items = tokenized_items[i:i + batch_size]
                current_batch_size = len(mini_batch_items)

                # Prepare tensors for the current mini-batch
                token_ids_list = [item['tokens'] for item in mini_batch_items]
                text_lens = torch.tensor([item['length'] for item in mini_batch_items], device=self.device)
                max_len = text_lens.max().item()

                sot_token = self.t3.hp.start_text_token
                eot_token = self.t3.hp.stop_text_token
                pad_token = eot_token

                text_tokens = torch.full((current_batch_size, max_len), pad_token, dtype=torch.long, device=self.device)
                for j, tokens in enumerate(token_ids_list):
                    text_tokens[j, :len(tokens)] = tokens

                text_tokens = F.pad(text_tokens, (1, 0), value=sot_token)
                text_tokens = F.pad(text_tokens, (0, 1), value=eot_token)
                text_lens += 2  # Account for SOT/EOT tokens

                if cfg_weight > 0.0:
                    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

                speech_tokens_batch, error_flags = self.t3.inference(
                    t3_cond=t3_cond_for_inference,
                    text_tokens=text_tokens,
                    text_token_lens=text_lens,
                    max_new_tokens=1000,  # TODO: use the value in config
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                    use_analyzer=use_analyzer,
                )
                all_speech_tokens_sorted.extend(speech_tokens_batch)
                all_error_flags_sorted.extend(error_flags)

        # Restore original order of results
        results_sorted = []
        for i, item in enumerate(tokenized_items):
            results_sorted.append({
                'original_index': item['original_index'],
                'speech_tokens': all_speech_tokens_sorted[i],
                'error_flag': all_error_flags_sorted[i],
            })
        results_sorted.sort(key=lambda x: x['original_index'])
        speech_tokens_batch_ordered = [r['speech_tokens'] for r in results_sorted]
        error_flags_ordered = [r['error_flag'] for r in results_sorted]

        # Offload T3 if requested, to free up VRAM for s3gen.
        if offload_t3 and self.device.type != 'cpu':
            self.t3.to('cpu')
        # Load s3gen back to GPU if it was offloaded
        if self.s3gen.device != self.device:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            self.s3gen.to(self.device)

        # Vocode speech tokens to audio
        wavs = []
        with torch.inference_mode():
            for speech_tokens, error_flag in zip(speech_tokens_batch_ordered, error_flags_ordered):
                if not skip_error_check and error_flag:
                    wavs.append(None) # Append None for error cases
                    continue

                speech_tokens = drop_invalid_tokens(speech_tokens)
            
                speech_tokens = speech_tokens[speech_tokens < 6561]

                speech_tokens = speech_tokens.to(self.device)

                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=active_conds.gen,
                )
                wav = wav.squeeze(0).detach().cpu().numpy()
                watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                wavs.append(torch.from_numpy(watermarked_wav).unsqueeze(0))

        return wavs if is_list_input else wavs[0]