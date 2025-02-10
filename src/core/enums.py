from enum import Enum


class SentimentEnum(str, Enum):
    BEARISH = 'Bearish'
    NEUTRAL = 'Neutral'
    BULLISH = 'Bullish'
