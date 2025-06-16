# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import os
import json

DEBUG_ANALYZER = os.getenv("DEBUG_ANALYZER", "0")  # Enable debug output for the analyzer

logger = logging.getLogger(__name__)

@dataclass
class AnalyzerAction:
    """The action to be taken by the generation loop based on the analyzer's state."""
    emit_eos: bool = False
    suppress_eos: bool = False
    error: bool = False
    replace_last_n_with_silence: Optional[int] = field(default=None)
    boost_silence: bool = False


@dataclass
class AnalyzerDebugStep:
    """Stores the complete state of the analyzer at a single frame."""
    frame_index: int
    text_position: int
    state_info: Dict[str, any]
    action: AnalyzerAction # The action recommended by the analyzer for this step.
    # --- New / Enhanced Debug Fields ---
    center_of_mass_pos: float
    logit_entropy: float
    health_score: float # The combined health score for this frame.
    component_scores: Dict[str, float] # The breakdown of the health score.
    # Logits and attention are detached and moved to CPU to prevent holding onto GPU memory
    raw_logits: torch.Tensor
    raw_attention_heads: torch.Tensor # Shape: (num_heads, S_kv)


@dataclass
class AnalyzerDebugLog:
    """A container for debug information across a full generation sequence."""
    analyzer_config: Dict[str, any]
    eos_idx: int
    steps: List[AnalyzerDebugStep] = field(default_factory=list)
    final_alignment_matrix: Optional[torch.Tensor] = None
    forced_stop_reason: Optional[str] = None

    def add_step(self, step_data: AnalyzerDebugStep):
        self.steps.append(step_data)

    def set_final_state(self, alignment_matrix: torch.Tensor, forced_stop_reason: str = "Natural Stop"):
        self.final_alignment_matrix = alignment_matrix
        self.forced_stop_reason = forced_stop_reason

    def _print_header_and_summary(self, forced_stop_reason: str):
        """Prints the report header, configuration, and summary."""
        print("\n" + "="*80)
        print("--- ALIGNMENT STREAM ANALYZER | DEBUG REPORT ---")
        print("="*80)
        print("\n[CONFIG]")
        for k, v in self.analyzer_config.items():
            print(f"- {k}: {v}")
        print(f"- EOS Token ID: {self.eos_idx}")

        print(f"\n[SUMMARY]")
        print(f"- Total Frames Generated: {len(self.steps)}")
        print(f"- Stop Reason: {self.forced_stop_reason or forced_stop_reason}")

    def _print_compact_analysis_table(self):
        """Prints a compact frame-by-frame overview of the analyzer's state."""
        print("\n" + "-"*80)
        print("[FRAME-BY-FRAME ANALYSIS]")
        print("-"*80)

        # Header
        header = f"{'Frame':<6} | {'Txt(argmax)':<12} | {'Txt(CoM)':<10} | {'Health':<8} | {'Entropy':<8} | {'Stagnation':<10} | {'Discont.':<10} | {'State':<8} | {'Action':<15}"
        print(header)
        print("-" * len(header))

        for step in self.steps:
            state_str = f"S:{step.state_info['Started'][0]} C:{step.state_info['Complete'][0]}"
            stagnation_str = f"{step.state_info['Stagnation']}"
            action_str = "None"
            if step.action.emit_eos:
                action_str = "EMIT_EOS"
            elif step.action.suppress_eos:
                action_str = "SUPPRESS_EOS"
            if step.action.replace_last_n_with_silence is not None:
                action_str = f"REPLACE({step.action.replace_last_n_with_silence})"
            if step.action.boost_silence:
                action_str += "+BOOST_SIL"

            row = (
                f"{step.frame_index:<6} | "
                f"{step.text_position:<12} | "
                f"{step.center_of_mass_pos:<10.2f} | "
                f"{step.health_score:<8.2f} | "
                f"{step.logit_entropy:<8.2f} | "
                f"{stagnation_str:<10} | "
                f"{step.state_info['Discont.']:<10} | "
                f"{state_str:<8} | "
                f"{action_str:<15}"
            )
            print(row)

    def _print_health_score_table(self):
        """Prints a detailed breakdown of the health score calculation at each frame."""
        print("\n" + "-"*80)
        print("[HEALTH SCORE BREAKDOWN]")
        print("-"*80)

        header = f"{'Frame':<5} | {'Health':<7} | {'FocusLoss':<10} | {'FocusBoost':<10} | {'Stagnation':<11} | {'EOS Drop':<10} | {'Long Tail':<10} | {'Repetition':<10}"
        print(header)
        print("-" * len(header))

        for step in self.steps:
            s = step.component_scores
            boost_active_str = "YES" if step.state_info.get("FocusBoostActive", False) else "NO"
            row = (
                f"{step.frame_index:<5} | "
                f"{step.health_score:<7.2f} | "
                f"{s.get('focus_loss', 0.0):<10.2f} | "
                f"{boost_active_str:<10} | "
                f"{s.get('stagnation', 0.0):<11.2f} | "
                f"{s.get('eos_drop', 0.0):<10.2f} | "
                f"{s.get('long_tail', 0.0):<10.2f} | "
                f"{s.get('repetition', 0.0):<10.2f}"
            )
            print(row)

    def _print_attention_head_table(self):
        """Prints a table showing the argmax and center of mass for each attention head at each frame."""
        if not self.steps or self.steps[0].raw_attention_heads is None:
            return
        print("\n" + "-"*80)
        print("[ATTENTION HEADS (Argmax / Center of Mass)]")
        print("-"*80)

        # Header
        num_heads = self.steps[0].raw_attention_heads.shape[0]
        col_width = 10  # Increased width for "argmax / com" format
        head_headers = [f"{f'Head {h_idx}':<{col_width}}" for h_idx in range(num_heads)]
        attn_table_header = f"{'Frame':<5} | " + " | ".join(head_headers)
        print(attn_table_header)
        print("-" * len(attn_table_header))

        # Table Rows
        for step in self.steps:
            row_parts = [f"{step.frame_index:<5}"]
            token_indices = torch.arange(step.raw_attention_heads.shape[-1], dtype=step.raw_attention_heads.dtype)
            for head_attn in step.raw_attention_heads:
                # Argmax
                argmax_pos = head_attn.argmax().item()

                # Center of Mass
                sum_of_weights = torch.sum(head_attn)
                if sum_of_weights > 1e-9:
                    com = (torch.sum(head_attn * token_indices) / sum_of_weights).item()
                    com_str = f"{com:.1f}"
                else:
                    com_str = 'nan'

                val_str = f"{argmax_pos} / {com_str}"
                row_parts.append(f"{val_str:<{col_width}}")
            print(" | ".join(row_parts))

    def _print_logits_table(self):
        """Prints a table showing the top-5 predicted tokens, their probabilities, and raw logits for each frame."""
        # This table is very wide, so we use a wider separator.
        header_text = "[PER-FRAME LOGITS (ID / Probability / Raw Logit)]"
        print("\n" + "-"*len(header_text))
        print(header_text)
        print("-"*len(header_text))

        # Header with increased width for logit values
        col_width = 26
        top_k_headers = [f"{f'Top-{i+1}':<{col_width}}" for i in range(5)]
        logits_header = f"{'Frame':<6} | " + " | ".join(top_k_headers) + f" | {'EOS':<22}"
        print(logits_header)
        print("-" * len(logits_header))

        # Table Rows
        for step in self.steps:
            raw_logits = step.raw_logits.squeeze()
            raw_probs = torch.softmax(raw_logits, dim=-1)

            # Get top 5 logits and their corresponding probabilities
            top_k_logits, top_k_indices = torch.topk(raw_logits, 5)
            top_k_probs = raw_probs[top_k_indices]

            # Format top-k strings
            top_k_strs = [
                f"{top_k_indices[i].item()} / {top_k_probs[i].item():.4f} / {top_k_logits[i].item():6.2f}"
                for i in range(5)
            ]
            
            # Format EOS string
            eos_prob = raw_probs[self.eos_idx].item()
            eos_logit = raw_logits[self.eos_idx].item()
            eos_str = f"{eos_prob:.5f} / {eos_logit:6.2f}"

            # Assemble and print the row
            row_parts = [f"{step.frame_index:<6}"]
            row_parts.extend([f"{s:<{col_width}}" for s in top_k_strs])
            row_parts.append(f"{eos_str:<22}")
            print(" | ".join(row_parts))

    def _print_final_alignment_heatmap(self):
        """Prints a character-based heatmap of the final alignment matrix, if available."""
        if self.final_alignment_matrix is None:
            return

        print("\n" + "-"*80)
        print("[FINAL ALIGNMENT MATRIX (T_audio, S_text)]")
        A = self.final_alignment_matrix
        heatmap_chars = ' .,:;o*#@'
        max_val = A.max()
        normalized_A = A / max_val if max_val > 0 else A
        header = "     " + "".join([f"{i:<3}" for i in range(A.shape[1])])
        print(header)
        print("    " + "-" * len(header))
        for i, row in enumerate(normalized_A):
            row_str = f"{i:<3} |"
            for val in row:
                char_index = int(val * (len(heatmap_chars) - 1))
                row_str += f" {heatmap_chars[char_index]} "
            print(row_str)
        print("-" * len(row_str))

    def print_report(self, forced_stop_reason: str = "Natural Stop"):
        """Prints a comprehensive, step-by-step debug report for the entire generation."""
        self._print_header_and_summary(forced_stop_reason)

        if not self.steps:
            print("\nNo steps to report.")
            print("="*80 + "\n")
            return

        self._print_compact_analysis_table()
        self._print_health_score_table()
        self._print_logits_table()
        self._print_attention_head_table()
        self._print_final_alignment_heatmap()

        print("\n" + "="*80)
        print("--- END OF REPORT ---")
        print("="*80 + "\n")


class AlignmentStreamAnalyzer:
    """
    Monitors the alignment stream during autoregressive generation to detect and
    prevent common failure modes, ensuring robust and high-quality synthesis.

    This analyzer operates on a frame-by-frame basis, assessing the model's
    attention and output logits to identify signs of instability. It combines
    multiple checks into a unified "health score" and uses hard-coded guardrails
    to stop generation when problems are detected.
    """
    # --- General Configuration ---
    SILENCE_TOKEN_ID = 4299
    ALIGNMENT_HEAD_IDX = 2            # The attention head primarily responsible for alignment.
    COMPLETION_TOKENS_FROM_END = 3    # How many tokens from the end of the text to consider "complete".

    # --- Health Score System ---
    # The health score starts at 1.0 and decreases as penalties for various issues
    # are applied. Generation stops if the score falls below HEALTH_SCORE_THRESHOLD.
    # Individual penalty components are normalized to their threshold (a component score of
    # 1.0 means the issue has reached its defined threshold, contributing its full
    # weight to the total penalty).
    HEALTH_SCORE_THRESHOLD = 0.0
    # Weights for each component's contribution to the total penalty
    W_FOCUS_LOSS = 0.5                  # How much CoM/argmax divergence contributes.
    W_FOCUS_LOSS_POST_COMPLETION = 0.8  # Boosted weight when focus loss remains high over several frames after completion.
    W_STAGNATION = 0.5                  # How much getting stuck on a token contributes.
    W_EOS_DROP = 0.6                    # How much a drop in EOS probability after a peak contributes.
    W_REPETITION = 1.0                  # Contribution of attending to past tokens post-completion.
    W_LONG_TAIL = 1.0                   # Contribution of lingering on final tokens post-completion.

    # --- Focus Loss Detection (Argmax/CoM Divergence) ---
    # The divergence (in tokens) at which the focus_loss_score becomes 1.0.
    FOCUS_DIVERGENCE_THRESHOLD = 5.0
    # The divergence above which focus loss is counted as a hard discontinuity event.
    FOCUS_LOSS_AS_DISCONTINUITY_THRESHOLD = 5.0
    # Window and threshold for boosting focus loss weight post-completion.
    FOCUS_LOSS_BOOST_WINDOW = 3                  # Check cumulative loss over this many recent frames.
    FOCUS_LOSS_BOOST_THRESHOLD = 2.5              # Cumulative focus loss score to trigger weight boost.

    # --- Stagnation Detection ---
    # Number of consecutive frames on the same token to be considered fully stagnated (score=1.0).
    # e.g., 25 frames * 40ms/frame = 1 second.
    STAGNATION_FRAME_THRESHOLD = 25

    # --- EOS Confidence Drop Detection ---
    # The analyzer will only start tracking a drop after the EOS probability has peaked above this value.
    MIN_EOS_PROB_FOR_TRACKING = 0.1

    # --- Discontinuity-Based Hallucination Detection (Hard Guardrail) ---
    DISCONTINUITY_JUMP_FORWARD = 5       # Max allowed forward jump in text tokens per frame.
    DISCONTINUITY_JUMP_BACKWARD = 3      # Max allowed backward jump in text tokens per frame.
    # The max number of consecutive discontinuities allowed at the beginning of the text.
    DISCONTINUITY_THRESHOLD_START = 100
    # The max number of consecutive discontinuities allowed at the very end of the text.
    DISCONTINUITY_THRESHOLD_END = 15
    # Controls the steepness of the sigmoid transition for the dynamic threshold between start and end.
    DISCONTINUITY_SIGMOID_STEEPNESS = 10.0

    # --- Post-Completion Anomaly Detection (as Health Score components) ---
    # The cumulative attention value on a single token that normalizes the score to 1.0.
    LONG_TAIL_CUMULATIVE_ATTENTION_THRESHOLD = 10.0
    REPETITION_CUMULATIVE_ATTENTION_THRESHOLD = 20.0
    # The number of final text tokens to monitor for the "long tail" effect.
    LONG_TAIL_TOKENS_TO_CHECK = 3
    # The number of final tokens to ignore when checking for repetition.
    REPETITION_TOKENS_TO_IGNORE_FROM_END = 5

    # --- False Start Detection ---
    FALSE_START_MIN_FRAMES = 2                  # Min frames required to evaluate a start.
    # Number of initial/final text tokens to check for instability.
    FALSE_START_BEGIN_TOKENS_TO_CHECK = 4
    FALSE_START_END_TOKENS_TO_CHECK = 2
    # Attention thresholds for instability. High attention on end tokens or low attention
    # on beginning tokens suggests a bad start.
    FALSE_START_END_TOKEN_ATTN_THRESHOLD = 0.1
    FALSE_START_BEGIN_TOKEN_ATTN_THRESHOLD = 0.5
    # Failsafe: if we pass this many frames and have moved this far in text, force start.
    FALSE_START_FAILSAFE_FRAMES = 25
    FALSE_START_FAILSAFE_TEXT_POS = 5

    # --- Intervention Logic ---
    # When a forced stop occurs after completion, we replace trailing frames that had a
    # health score below this threshold with silence tokens.
    REPLACE_HEALTH_THRESHOLD = 0.3
    # If health drops below this threshold post-completion, start boosting the silence token probability.
    BOOST_SILENCE_HEALTH_THRESHOLD = 0.5
    SILENCE_BOOST_LOGIT_VALUE = 5.0 # Additive boost to the silence token's logit.

    @classmethod
    def load_config_from_json(cls, config_path: str):
        """
        Loads configuration from a JSON file and monkey-patches the class constants.
        This affects all subsequent instances of the analyzer.
        """
        if not os.path.exists(config_path):
            logger.warning(f"Analyzer config file not found at {config_path}. Using default values.")
            return

        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load or parse analyzer config from {config_path}: {e}. Using default values.")
            return

        class_constants = {k for k in dir(cls) if k.isupper() and not k.startswith('_')}
        loaded_count = 0
        for key, value in config_data.items():
            if key in class_constants:
                setattr(cls, key, value)
                loaded_count += 1
            else:
                logger.warning(f"Analyzer config: Ignoring unknown key '{key}' from {config_path}.")

        if loaded_count > 0:
            logger.info(f"Loaded and applied {loaded_count} configuration values from {config_path} to AlignmentStreamAnalyzer class.")

    @classmethod
    def save_config_to_json(cls, config_path: str):
        """
        Saves the current class constants to a JSON file.
        This is useful for creating a template configuration file.
        """
        class_constants = {k for k in dir(cls) if k.isupper() and not k.startswith('_')}
        config_data = {key: getattr(cls, key) for key in sorted(class_constants)}

        try:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            logger.info(f"Saved current AlignmentStreamAnalyzer configuration to {config_path}.")
        except IOError as e:
            logger.error(f"Failed to save analyzer config to {config_path}: {e}")

    def __init__(self, queue, text_tokens_slice, eos_idx=0, text_len: Optional[int] = None):
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.text_len = text_len if text_len is not None else (j - i)
        self.eos_idx = eos_idx

        # State tracking
        self.alignment = torch.zeros(0, j - i) # Stores the history of the alignment head's attention
        self.curr_frame_pos = 0
        self.text_position = 0
        self.started = False
        self.started_at = None
        self.complete = False
        self.completed_at = None
        self.consecutive_discontinuities = 0
        self.dynamic_threshold = self.DISCONTINUITY_THRESHOLD_START

        # Health Score State
        self.health_score = 1.0
        self.component_scores = {}
        self.health_score_history: List[float] = []
        self.focus_loss_history: List[float] = []
        self.focus_loss_boost_active = False
        self.last_argmax_pos = -1
        self.stagnation_counter = 0
        self.max_eos_prob_seen = 0.0

        self.debug_log = self._init_debug_log()

    def _init_debug_log(self) -> Optional[AnalyzerDebugLog]:
        if DEBUG_ANALYZER != "1":
            return None
        # Dynamically get all uppercase config attributes from the class
        # This will correctly pick up the default or monkey-patched values.
        config = {k: getattr(self, k) for k in dir(self) if k.isupper() and not k.startswith('_')}
        return AnalyzerDebugLog(analyzer_config=config, eos_idx=self.eos_idx)

    # --- Helper Methods for Detection Logic ---

    def _check_discontinuity(self, current_pos: int, prev_pos: int) -> bool:
        """
        Checks if the attention has jumped an implausible distance between frames.

        Args:
            current_pos: The current text token index (argmax of attention).
            prev_pos: The text token index from the previous frame.

        Returns:
            True if the jump distance is outside the configured safe limits.
        """
        jump_dist = current_pos - prev_pos
        return not (-self.DISCONTINUITY_JUMP_BACKWARD < jump_dist < self.DISCONTINUITY_JUMP_FORWARD)

    def _check_discontinuity_timeout(self, progress: float) -> bool:
        """
        Checks if there have been too many consecutive discontinuities.

        The allowed number of discontinuities is dynamic, starting high and decreasing
        as generation progresses to become stricter towards the end.

        Args:
            progress: The generation progress as a float from 0.0 to 1.0.

        Returns:
            True if the number of consecutive discontinuities exceeds the dynamic threshold.
        """
        # Calculate a dynamic threshold that gets stricter as generation progresses.
        sigmoid_input = self.DISCONTINUITY_SIGMOID_STEEPNESS * (progress - 0.5)
        sigmoid_value = torch.sigmoid(torch.tensor(sigmoid_input)).item()
        self.dynamic_threshold = self.DISCONTINUITY_THRESHOLD_START - \
            (self.DISCONTINUITY_THRESHOLD_START - self.DISCONTINUITY_THRESHOLD_END) * sigmoid_value
        return self.consecutive_discontinuities > self.dynamic_threshold

    def _check_false_start(self, A: torch.Tensor, T: int) -> bool:
        """
        Checks for an unstable start, e.g., attention on end tokens or weak initial attention.

        A "false start" is when the model fails to lock onto the beginning of the
        text, often indicated by spurious attention on later parts of the text or a
        failure to put significant weight on the first few tokens.

        Args:
            A: The alignment history matrix (T_audio, S_text).
            T: The current number of generated frames.

        Returns:
            True if the start is deemed unstable ("false").
        """
        if T < self.FALSE_START_MIN_FRAMES:
            return True # Not enough frames to judge, assume false start.

        # Unstable if attention is high on the final tokens.
        attn_on_end = A[-self.FALSE_START_MIN_FRAMES:, -self.FALSE_START_END_TOKENS_TO_CHECK:].max()
        end_is_unstable = attn_on_end > self.FALSE_START_END_TOKEN_ATTN_THRESHOLD

        # Unstable if attention fails to lock onto the beginning tokens.
        attn_on_begin = A[:, :self.FALSE_START_BEGIN_TOKENS_TO_CHECK].max()
        begin_is_unstable = attn_on_begin < self.FALSE_START_BEGIN_TOKEN_ATTN_THRESHOLD
        
        # Failsafe to prevent getting stuck in a false start state indefinitely.
        if T > self.FALSE_START_FAILSAFE_FRAMES and self.text_position > self.FALSE_START_FAILSAFE_TEXT_POS:
            logger.warning("Forcing 'started' state due to advanced position despite instability flags.")
            return False
        
        return end_is_unstable or begin_is_unstable

    def _check_completion(self) -> bool:
        """
        Checks if the alignment has reached the end of the text.

        Returns:
            True if the current text position is within the configured completion zone.
        """
        return self.text_position >= self.text_len - self.COMPLETION_TOKENS_FROM_END

    # --- Main Analysis Step ---

    def step(self, logits: torch.Tensor, last_attention_heads: torch.Tensor) -> AnalyzerAction:
        """
        Analyzes the current generation step and returns an action for the generation loop.
        """
        # --- 1. Pre-computation and State Update ---
        i, j = self.text_tokens_slice

        # last_attention_heads has shape (num_heads, L_query, L_key).
        # We only care about the attention from the *last* query token.
        last_token_attention_heads = last_attention_heads[:, -1, :]  # Shape: (num_heads, L_key)

        # Slice to get the attention over the relevant text tokens.
        last_token_attention_heads_sliced = last_token_attention_heads[:, i:j]  # Shape: (num_heads, text_len_slice)

        # Get the clean alignment signal from the designated head
        alignment_attn_row = last_token_attention_heads_sliced[self.ALIGNMENT_HEAD_IDX, :].clone().cpu()

        # The main alignment history `A` now stores the history of the specific alignment head.
        A_chunk_head2 = alignment_attn_row.unsqueeze(0)
        self.alignment = torch.cat((self.alignment, A_chunk_head2), dim=0)
        A = self.alignment
        T, S = A.shape


        # --- 2. Positional and Discontinuity Analysis  ---
        cur_text_posn = alignment_attn_row.argmax().item()
        token_indices = torch.arange(len(alignment_attn_row), device=alignment_attn_row.device)
        com_pos = (torch.sum(alignment_attn_row * token_indices) / (alignment_attn_row.sum() + 1e-9)).item()
        focus_divergence = abs(cur_text_posn - com_pos)

        is_jump_discontinuous = self._check_discontinuity(cur_text_posn, self.text_position)
        has_significant_focus_loss = focus_divergence > self.FOCUS_LOSS_AS_DISCONTINUITY_THRESHOLD
        discontinuity_events_this_step = int(is_jump_discontinuous) + int(has_significant_focus_loss)

        if discontinuity_events_this_step > 0:
            self.consecutive_discontinuities += discontinuity_events_this_step
        else:
            self.consecutive_discontinuities = 0

        # Only update text position if there wasn't a jump. Focus loss alone doesn't prevent progress.
        if not is_jump_discontinuous:
            self.text_position = cur_text_posn

        progress = min(1.0, self.text_position / (self.text_len - 1 + 1e-6))
        discontinuity_timeout = self._check_discontinuity_timeout(progress)

        # --- 3. Start and Completion State Update ---
        if not self.started:
            is_false_start = self._check_false_start(A, T)
            self.started = not is_false_start
            if self.started and self.started_at is None:
                self.started_at = T

        if not self.complete:
            self.complete = self._check_completion()
            if self.complete and self.completed_at is None:
                self.completed_at = T

        # --- 4. Calculate Health Score Components ---
        self.component_scores = {}

        # 4a. Focus Loss Score (Argmax/CoM Divergence)
        focus_loss_score = min(1.0, focus_divergence / self.FOCUS_DIVERGENCE_THRESHOLD)
        self.component_scores['focus_loss'] = focus_loss_score
        self.focus_loss_history.append(focus_loss_score)

        # 4b. Stagnation Score
        if not self.complete and cur_text_posn == self.last_argmax_pos and self.started:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.last_argmax_pos = cur_text_posn
        stagnation_score = min(1.0, self.stagnation_counter / self.STAGNATION_FRAME_THRESHOLD)
        self.component_scores['stagnation'] = stagnation_score

        # 4c. EOS Confidence Drop Score
        current_eos_prob = torch.softmax(logits, dim=-1).squeeze()[self.eos_idx].item()
        self.max_eos_prob_seen = max(self.max_eos_prob_seen, current_eos_prob)
        eos_drop_score = 0.0
        if self.max_eos_prob_seen > self.MIN_EOS_PROB_FOR_TRACKING:
            drop_ratio = (self.max_eos_prob_seen - current_eos_prob) / self.max_eos_prob_seen
            eos_drop_score = max(0.0, drop_ratio)
        self.component_scores['eos_drop'] = eos_drop_score
        self.component_scores['raw_max_eos_prob'] = self.max_eos_prob_seen # For debug

        # 4d. Post-Completion Scores (Long Tail & Repetition)
        long_tail_score, repetition_score = 0.0, 0.0
        if self.complete and self.completed_at is not None:
            attn_post_completion = A[self.completed_at:]
            # Long Tail
            attn_on_final_tokens = attn_post_completion[:, -self.LONG_TAIL_TOKENS_TO_CHECK:]
            cumulative_attn_on_final = attn_on_final_tokens.sum(dim=0)
            long_tail_raw_val = cumulative_attn_on_final.max().item() if cumulative_attn_on_final.numel() > 0 else 0.0
            long_tail_score = long_tail_raw_val / self.LONG_TAIL_CUMULATIVE_ATTENTION_THRESHOLD
            # Repetition
            attn_on_past_tokens = attn_post_completion[:, :-self.REPETITION_TOKENS_TO_IGNORE_FROM_END]
            if attn_on_past_tokens.numel() > 0:
                repetition_raw_val = attn_on_past_tokens.max(dim=1).values.sum().item()
                repetition_score = repetition_raw_val / self.REPETITION_CUMULATIVE_ATTENTION_THRESHOLD
        self.component_scores['long_tail'] = long_tail_score
        self.component_scores['repetition'] = repetition_score

        # --- 5. Calculate Final Health Score and Formulate Action ---
        w_focus = self.W_FOCUS_LOSS
        if self.complete:
            # Check if sustained focus loss should trigger a permanent boost.
            if not self.focus_loss_boost_active and sum(self.focus_loss_history[-self.FOCUS_LOSS_BOOST_WINDOW:]) > self.FOCUS_LOSS_BOOST_THRESHOLD:
                self.focus_loss_boost_active = True
            
            # Apply the boosted weight if the latch is active.
            if self.focus_loss_boost_active:
                w_focus = self.W_FOCUS_LOSS_POST_COMPLETION
        
        # Calculate the total penalty from all active components.
        total_penalty = (w_focus * focus_loss_score +
                         self.W_STAGNATION * stagnation_score)

        if self.complete:
            total_penalty += (self.W_EOS_DROP * eos_drop_score +
                              self.W_REPETITION * repetition_score +
                              self.W_LONG_TAIL * long_tail_score)
        
        # Health score starts at 1.0 and decreases as penalties accumulate.
        self.health_score = 1.0 - total_penalty

        is_forced_stop = (self.health_score < self.HEALTH_SCORE_THRESHOLD) or discontinuity_timeout

        # Suppress EOS unless we are complete or forcing a stop.
        suppress_eos = not self.complete and not is_forced_stop

        # Boost silence probability if health is deteriorating post-completion.
        boost_silence = False
        if self.complete and self.health_score < self.BOOST_SILENCE_HEALTH_THRESHOLD:
            boost_silence = True

        action = AnalyzerAction(
            emit_eos=is_forced_stop,
            suppress_eos=suppress_eos,
            error=is_forced_stop,
            boost_silence=boost_silence,
        )

        # Special handling for forced stop after completion: replace bad frames at the end with silence.
        if is_forced_stop and self.complete:
            action.error = False # Not a fatal error, but a controlled stop.
            n_to_replace = 0
            for score in reversed(self.health_score_history):
                if score < self.REPLACE_HEALTH_THRESHOLD:
                    n_to_replace += 1
                else:
                    break
            if n_to_replace > 0:
                action.replace_last_n_with_silence = n_to_replace

        self.health_score_history.append(self.health_score)

        # --- 6. Save Debug State (if enabled) ---
        if self.debug_log:
            self._save_debug_step(
                raw_logits=logits,
                raw_attention_heads=last_token_attention_heads_sliced,
                center_of_mass_pos=com_pos,
                action=action,
            )

        self.curr_frame_pos += 1
        return action

    def _save_debug_step(self, **kwargs):
        """Helper to compute debug metrics and save the step to the log."""
        # Calculate Logit Entropy for general info
        probs = torch.softmax(kwargs['raw_logits'], dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=-1).item()

        state_info = {
            "Complete": "YES" if self.complete else "no",
            "Started": "YES" if self.started else "no",
            "Discont.": f"{self.consecutive_discontinuities}/{self.dynamic_threshold:.1f}",
            "Stagnation": f"{self.stagnation_counter}/{self.STAGNATION_FRAME_THRESHOLD}",
            "FocusBoostActive": self.focus_loss_boost_active,
        }
        debug_step = AnalyzerDebugStep(
            frame_index=self.curr_frame_pos,
            text_position=self.text_position,
            center_of_mass_pos=kwargs['center_of_mass_pos'],
            logit_entropy=entropy,
            health_score=self.health_score,
            component_scores=self.component_scores.copy(),
            state_info=state_info,
            raw_logits=kwargs['raw_logits'].clone().cpu().detach(),
            raw_attention_heads=kwargs['raw_attention_heads'].clone().cpu().detach(),
            action=kwargs['action']
        )
        self.debug_log.add_step(debug_step)

        if kwargs['action'].error or kwargs['action'].emit_eos:
            reason = "Natural Stop"
            if self.consecutive_discontinuities > self.dynamic_threshold:
                reason = f"Discontinuity timeout ({self.consecutive_discontinuities} > {self.dynamic_threshold:.1f})."
            elif self.health_score < self.HEALTH_SCORE_THRESHOLD:
                reason = f"Health score {self.health_score:.2f} fell below threshold {self.HEALTH_SCORE_THRESHOLD}."

            # Set final state only once
            if self.debug_log.forced_stop_reason is None:
                self.debug_log.set_final_state(self.alignment, forced_stop_reason=reason)