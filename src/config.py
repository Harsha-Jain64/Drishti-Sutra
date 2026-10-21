"""All tunable thresholds and scoring weights for Drishti-Sutra.

Constants are heuristic aids to flag potential anomalies, not calibrated
statistical probabilities or legal thresholds.
"""

# Entity resolution
NAME_MATCH_THRESHOLD: int = 85          # rapidfuzz token_set_ratio for 'suspected' link
NAME_CONFIRMED_THRESHOLD: int = 100     # exact normalised token match

# Time windows (in minutes)
MESSAGE_TIME_WINDOW_MINUTES: int = 2    # align screenshot vs export messages within this window
LOG_TIME_WINDOW_MINUTES: int = 5        # log event vs chat message proximity
POST_EVENT_MODIFICATION_WINDOW_MINUTES: int = 15  # file-mod warning after a transaction

# Text similarity
TEXT_SIMILARITY_THRESHOLD: int = 80     # rapidfuzz token_set_ratio for message-text alignment

# Base confidence scores per contradiction type (heuristic)
BASE_SCORE_DOC_FIELD_MISMATCH: float = 0.85
BASE_SCORE_QR_PRINTED: float = 0.90
BASE_SCORE_AMOUNT_MISMATCH: float = 0.88
BASE_SCORE_MESSAGE_ABSENT: float = 0.80
BASE_SCORE_ORDER_INVERSION: float = 0.85
BASE_SCORE_LOG_CONFLICT: float = 0.82

# Score adjustments
PENALTY_INFERRED_DATE: float = 0.08    # date inferred from companion source
PENALTY_OCR_EXTRACTION: float = 0.05   # text came from OCR (higher uncertainty)
PENALTY_SUSPECTED_ENTITY: float = 0.07 # entity link is fuzzy, not confirmed
CORROBORATION_BONUS: float = 0.06      # a third independent source agrees

# Timeline
ANOMALOUS_GAP_MINUTES: int = 60        # gaps above this are flagged on the timeline

# ELA
ELA_STD_VARIANCE_THRESHOLD: float = 10.0  # std block variance threshold for ELA finding
