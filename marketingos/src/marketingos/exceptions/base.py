"""
Base exception hierarchy for MarketingOS.

All custom exceptions in the application should inherit from
MarketingOSError to provide a consistent exception hierarchy.
"""

from __future__ import annotations


class MarketingOSError(Exception):
    """Base exception for the entire MarketingOS application."""
