import pandas as pd
from dataclasses import dataclass
from typing import Optional, List
from sklearn.metrics import confusion_matrix, classification_report
from transformers import TextClassificationPipeline

from src.utils.preprocessing import clean
from src.inference.inferencer_utils import predict_sentiment
from src.core.schemas import SentimentAnalysis, ClassificationReport, BatchSentimentAnalysis


@dataclass
class InferenceConfig:
    max_length: int = 64
    truncation: bool = True
    padding: str = "max_length"
    return_all_scores: bool = True


class SentimentInference:
    def __init__(self, model, tokenizer, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.pipe = self._create_pipeline(model, tokenizer)
        self.label_mapper = {
            0: 'Bearish',
            1: 'Neutral',
            2: 'Bullish'
        }

    def _create_pipeline(self, model, tokenizer) -> TextClassificationPipeline:
        return TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_all_scores=self.config.return_all_scores
        )

    def _prepare_data(self, data: pd.DataFrame) -> List[str]:
        """Prepare text data for inference."""
        data["clean_text"] = data["text"].apply(clean)
        return data["clean_text"].tolist()

    def _generate_predictions(self, texts: List[str]) -> pd.DataFrame:
        """Generate predictions for the given texts."""
        predictions = predict_sentiment(texts, self.pipe)
        return pd.DataFrame(predictions)

    def _generate_report(self, true_labels: List[str], predicted_labels: List[str]):
        """Generate and print classification metrics."""
        cm = confusion_matrix(true_labels, predicted_labels)
        cls_report = classification_report(true_labels, predicted_labels)
        return cm, cls_report

    def inference_from_file(self, test_path: str) -> BatchSentimentAnalysis:
        """Run inference on data from a file and optionally save results."""

        test_df = pd.read_csv(test_path)
        predictions = self._run_inference(test_df)
        report = ClassificationReport()

        if 'label' in test_df:
            true_labels = [self.label_mapper[label] for label in test_df["label"].tolist()]
            predicted_labels = [pred.label.value for pred in predictions]
            cm, cls_report = self._generate_report(true_labels, predicted_labels)
            report = ClassificationReport(confusion_matrix=cm, classification_report=cls_report)

        return BatchSentimentAnalysis(sentiments=predictions, report=report)


    def inference_from_text(self, text: str) -> pd.DataFrame:
        """Run inference on a single text input."""
        test_df = pd.DataFrame({'text': [text]})
        return self._run_inference(test_df)

    def _run_inference(self, data: pd.DataFrame) -> List[SentimentAnalysis]:
        texts = self._prepare_data(data)
        pred_df = self._generate_predictions(texts)
        sentiment_analysis_list = [
            SentimentAnalysis(label=row['predicted_label'], sentiment_index=row['sentiment_index'])
            for index, row in pred_df.iterrows()
        ]
        return sentiment_analysis_list
