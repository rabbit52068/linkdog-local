"""Streaming sentence segmentation for low-latency voice playback.

Mirrors the original xiaozhi-esp32-server `_get_segment_text` behaviour: feed
LLM token deltas in, get complete sentences out as soon as a punctuation
boundary is reached. The first sentence uses a looser punctuation set (commas
included) so the first audible chunk starts as early as possible; subsequent
sentences use a stricter set to avoid chopping mid-thought.
"""

from __future__ import annotations

from typing import List, Optional

# First sentence: cut on commas too, so the first sound comes out fast.
FIRST_SENTENCE_PUNCTUATIONS = (
    "，", ",", "~", "、", "。", ".", "？", "?", "！", "!", "；", ";", "：", ":",
)
# Subsequent sentences: only cut on strong boundaries.
PUNCTUATIONS = (
    "。", ".", "？", "?", "！", "!", "；", ";", "：", ":",
)


class SentenceSplitter:
    """Accumulate streaming text and emit complete sentences on punctuation."""

    def __init__(
        self,
        first_punctuations: tuple = FIRST_SENTENCE_PUNCTUATIONS,
        punctuations: tuple = PUNCTUATIONS,
    ) -> None:
        self._first_punctuations = first_punctuations
        self._punctuations = punctuations
        self._buffer = ""
        self._is_first = True

    def feed(self, text: str) -> List[str]:
        """Append a token delta and return any complete sentences it forms."""
        if not text:
            return []
        self._buffer += text
        return self._drain()

    def flush(self) -> Optional[str]:
        """Return any remaining unsegmented text (end of stream)."""
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            return remaining
        return None

    def _drain(self) -> List[str]:
        punctuations = (
            self._first_punctuations if self._is_first else self._punctuations
        )
        sentences: List[str] = []
        while True:
            cut = self._find_cut(punctuations)
            if cut < 0:
                break
            segment = self._buffer[: cut + 1]
            self._buffer = self._buffer[cut + 1 :]
            cleaned = segment.strip()
            if cleaned:
                sentences.append(cleaned)
            if self._is_first:
                self._is_first = False
                punctuations = self._punctuations
        return sentences

    def _find_cut(self, punctuations: tuple) -> int:
        # Earliest punctuation position wins: cut the first complete sentence.
        earliest = -1
        for punct in punctuations:
            pos = self._buffer.find(punct)
            if pos != -1 and (earliest == -1 or pos < earliest):
                earliest = pos
        return earliest
