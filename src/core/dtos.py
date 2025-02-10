from typing import Optional, List
from pydantic import BaseModel
from src.core.schemas import SentimentAnalysis, ClassificationReport, AspectBasedSentiments


class TweetRequestDTO(BaseModel):
    text: str


class BaseResponse(BaseModel):
    status: str
    error_message: Optional[str] = None


class SentimentResponseDTO(BaseResponse):
    sentiment_analysis: SentimentAnalysis = None


class BatchSentimentResponseDTO(BaseResponse):
    sentiment_analysis: List[SentimentAnalysis] = None
    report: ClassificationReport = None


class ABSAResponseDTO(BaseResponse):
    sentiment_analysis: AspectBasedSentiments = None


class BatchABSAResponseDTO(BaseResponse):
    sentiment_analysis: List[AspectBasedSentiments] = None
