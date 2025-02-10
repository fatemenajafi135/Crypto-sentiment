
from src.core.schemas import AspectBasedSentiments, SentimentAnalysis, BatchSentimentAnalysis
from src.inference.sentiment_inference import SentimentInference


def get_sentiment(tweet: str, model, tokenizer) -> SentimentAnalysis:
    inference_engine = SentimentInference(model, tokenizer)
    result = inference_engine.inference_from_text(tweet)
    return result


def get_sentiment_file(test_path, model, tokenizer) -> BatchSentimentAnalysis:

    inference_engine = SentimentInference(model, tokenizer)
    results = inference_engine.inference_from_file(
        test_path=test_path,
    )
    return results
