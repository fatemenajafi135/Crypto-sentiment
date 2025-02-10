from typing import List
from pydantic import BaseModel, Field

from src.core.enums import SentimentEnum


class SentimentAnalysis(BaseModel):
    label: SentimentEnum
    sentiment_index: float


class ClassificationReport(BaseModel):
    confusion_matrix: List[List[int]] = None
    classification_report: str = None


class BatchSentimentAnalysis(BaseModel):
    sentiments: List[SentimentAnalysis]
    report: ClassificationReport = None


class CoinBasedSentiment(BaseModel):
    coin: str = Field(..., description="Name of the cryptocurrency")
    sentiment: SentimentEnum


class AspectBasedSentiments(BaseModel):
    coins: List[CoinBasedSentiment] = Field(..., description="List of coins and their specific sentiments")
    overall_sentiment: SentimentEnum

