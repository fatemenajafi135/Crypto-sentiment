# LM
# -----------------------------------
BASE_MODEL = "ElKulako/cryptobert"
MODEL_NAME = "ElKulako/cryptobert"
NUM_LABEL = 3
SAVE_MODEL_PATH = ''


# LLM
# -----------------------------------
LLM_NAME = "gpt-4o-mini-2024-07-18"
TEMP = 0.1
PROMPT = """You are a financial analyst specializing in cryptocurrency markets.
Analyze the tweet to identify all mentions of cryptocurrencies and determine the associated sentiment for each.
Additionally, determine the overall sentiment of the tweet."""
