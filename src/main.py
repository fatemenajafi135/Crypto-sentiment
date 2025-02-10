import time
from pathlib import Path
from fastapi import FastAPI, Request, Depends
from fastapi import File, UploadFile

from src.inference.inferencer_utils import load_model, load_tokenizer
from src.routers.absa_router import get_aspect_based_sentiment, get_aspect_based_sentiment_file
from src.routers.sa_router import get_sentiment, get_sentiment_file
from src.core.dtos import (
    TweetRequestDTO,
    SentimentResponseDTO,
    BatchSentimentResponseDTO,
    ABSAResponseDTO,
    BatchABSAResponseDTO
)


test_file_directory = Path("./test_files")
app = FastAPI()


@app.on_event("startup")
def startup_event():
    start_time = time.time()
    app.state.model = load_model()
    app.state.tokenizer = load_tokenizer()
    end_time = time.time()
    print(f'Model loaded in {int(end_time - start_time)} seconds.')


def get_model(request: Request):
    return request.app.state.model


def get_tokenizer(request: Request):
    return request.app.state.tokenizer


async def upload_file(file):
    test_file_directory.mkdir(parents=True, exist_ok=True)
    file_path = test_file_directory / file.filename
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    return file_path


@app.post("/predict_sentiment", response_model=SentimentResponseDTO)
def predict_sentiment(
        tweet_request: TweetRequestDTO,
        model=Depends(get_model),
        tokenizer=Depends(get_tokenizer)
):
    try:
        prediction = get_sentiment(tweet_request.text, model=model, tokenizer=tokenizer)
        return SentimentResponseDTO(sentiment_analysis=prediction[0], status="success")
    except Exception as e:
        print(e)
        return SentimentResponseDTO(status="error", error_message=str(e))


@app.post('/predict_sentiment_file', response_model=BatchSentimentResponseDTO)
async def predict_sentiment_file(
        file: UploadFile = File(...),
        model=Depends(get_model),
        tokenizer=Depends(get_tokenizer)
):
    try:
        file_path = await upload_file(file)
        batch_sentiments = get_sentiment_file(file_path, model=model, tokenizer=tokenizer)
        return BatchSentimentResponseDTO(sentiment_analysis=batch_sentiments.sentiments, report=batch_sentiments.report, status="success")
    except Exception as e:
        return BatchSentimentResponseDTO(status="error", error_message=str(e))


@app.post('/predict_aspect_based_sentiment', response_model=ABSAResponseDTO)
def predict_aspect_based_sentiment(tweet_request: TweetRequestDTO):
    response = get_aspect_based_sentiment(
        tweet=tweet_request.text,
    )
    try:
        return ABSAResponseDTO(sentiment_analysis=response, status="success")
    except Exception as e:
        return ABSAResponseDTO(status="error", error_message=str(e))


@app.post('/predict_aspect_based_sentiment_file', response_model=BatchABSAResponseDTO)
async def predict_aspect_based_sentiment_file(
        file: UploadFile = File(...),
):
    try:
        file_path = await upload_file(file)
        response = get_aspect_based_sentiment_file(file_path)
        return BatchABSAResponseDTO(sentiment_analysis=response, status="success")
    except Exception as e:
        return BatchABSAResponseDTO(status="error", error_message=str(e))


@app.get('/')
def root():
    return {"response": 'Welcome to Sentiment Analysis on Crypto Tweets!'}
