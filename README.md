# VADER + BERT Sentiment Analysis (Streamlit)

A simple Streamlit app to analyze sentiment using **VADER** and **BERT (DistilBERT SST-2)** with separate sections and charts.

## Features
- Single-text and CSV modes
- Separate VADER and BERT sections (counts, pie charts, common words)
- Download analyzed CSV

## Requirements
See `requirements.txt` (works on Streamlit Cloud and locally).

## Quick Start (Local)
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## CSV Format
Your CSV must contain a column named **`reviewText`**. See `amazon_reviews_sample.csv` for reference.

## Notes
- BERT requires `transformers` and `torch`. If not available, app still works with VADER.
- For long texts, input is truncated to fit DistilBERT max length (512 tokens).

  Created by zebgenai
