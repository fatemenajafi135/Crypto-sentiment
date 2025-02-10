import os
from typing import List
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from src.core.schemas import AspectBasedSentiments
import config

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


def get_aspect_based_sentiment(tweet: str) -> AspectBasedSentiments:
    client = OpenAI(api_key=api_key, )
    try:
        completion = client.beta.chat.completions.parse(
            model=config.LLM_NAME,
            messages=[
                {"role": "system", "content": config.PROMPT},
                {"role": "user", "content": tweet}
            ],
            response_format=AspectBasedSentiments,
        )

        sentiment = completion.choices[0].message.parsed
        return sentiment

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_aspect_based_sentiment_file(file_path: str) -> List[AspectBasedSentiments]:
    df = pd.read_csv(file_path)
    df['absa'] = df.text.apply(get_aspect_based_sentiment)
    return df['absa'].to_list()
