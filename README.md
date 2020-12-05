# Quora QA Chatbot

[report](https://github.com/sikfeng/quora-qa-chatbot/blob/master/Quora-Question-Answering-Chatbot-report.pdf) | [poster](https://github.com/sikfeng/quora-qa-chatbot/blob/master/Quora-Question-Answering-Chatbot-poster.pdf)

This is a Quora Question Answer Chatbot for Telegram.

The `test.csv` and `train.csv` files should be obtained from https://www.kaggle.com/quora/question-pairs-dataset.

The API key should be placed inside the file `API_KEY`.

## Running

You should first run `python3 precompute.py` to preprocess the data and compute the tf-idf vectors. They will be saved under `precompute/`.

To run the bot, run `python3 server.py`.
