# HIRE-MATCH-RESUME-PARSER

This project is an AI-powered resume screening tool that helps match resumes to job roles using NLP and machine learning.

## Features
- Extracts text from PDF and DOCX resumes
- Uses BERT embeddings and TF-IDF for feature extraction
- Predicts role suitability using trained models
- Streamlit web interface for easy use

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streamlit app:
   ```bash
   streamlit run newstreamlit.py
   ```

## Project Structure
- `newstreamlit.py`: Main Streamlit app
- `create_bert_model.py`: Script to create and save BERT model
- `rf_clf.pkl`, `tfidf.pkl`, `encoder.pkl`, `bert_embedder.pkl`: Model files
- `data/`: Contains sample resumes organized by job role
- `Resume/Resume.txt`: Example resume text
- `.gitignore`: Excludes unnecessary files from Git tracking

## License
MIT

## Author
Jyothsna2004
