"""Backward-compatible import for the project's shared text cleaner."""

try:
    from .text_processing import clean_text, ensure_nltk_resources
except ImportError:
    from text_processing import clean_text, ensure_nltk_resources


__all__ = ["clean_text", "ensure_nltk_resources"]
