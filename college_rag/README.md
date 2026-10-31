# College Knowledge Assistant (RAG)

An AI assistant that answers student questions **only** from official college
documents, shows the supporting sources, and refuses to answer when the
information is not available.

## Architecture

```
Documents (PDF/TXT) -> load -> chunk (120 words, 30 overlap)
        -> embed (all-MiniLM-L6-v2) -> index/kb.pkl
Question -> embed -> cosine top-k search
        -> similarity < threshold ?  -> "Not found" (LLM is never called)
        -> else build prompt with numbered context -> Ollama gemma:2b (temperature 0)
        -> answer + sources (file, page, last-updated date, score)
```

| Component | Choice | Reason |
|---|---|---|
| Loader | pypdf / plain text | Simple, no OCR needed for text PDFs |
| Chunking | 120 words, 30 overlap | One policy clause fits in a chunk; overlap avoids cutting sentences |
| Embeddings | all-MiniLM-L6-v2 | Small, fast, runs on CPU, handles paraphrases |
| Vector store | NumPy matrix + cosine | Few hundred chunks; no external DB needed |
| LLM | Ollama gemma:2b | Free and local; temperature 0 for stable answers |
| UI | Streamlit | Chat UI, source panel and upload in one file |

## Anti-hallucination design

1. **Retrieval threshold**: if the best similarity is below `MIN_SCORE`, the app refuses without calling the LLM.
2. **Strict prompt**: answer only from context, otherwise return a fixed "not found" sentence.
3. **Conflict rule**: if sources disagree, the model must report both values with file name and last-updated date.
4. **Source panel**: every answer lists file, page, date and similarity score.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt

# LLM (one time)
ollama pull gemma:2b

# Run
streamlit run app.py
```
The first run builds the index from `docs/`. Use the sidebar to upload more files and rebuild.

## Testing

```bash
python run_tests.py --no-llm   # retrieval-only check
python run_tests.py            # full pipeline (Ollama must be running)
```
12 test questions with expected sources/keywords are in `tests/test_cases.json`;
results are written to `tests/results.md`.

| Mandatory scenario | Test IDs |
|---|---|
| 1. Clear answer in one document | 1-4 |
| 2. Needs two documents | 5-6 |
| 3. Unrelated / absent | 7-9 |
| 4. Different wording | 10-11 |
| 5. Conflicting documents | 12 (handbook says 75%, circular says 80%) |

## Technical problem faced (fill in with your own experience)
Example: related-but-absent questions (test 9) scored above the threshold because
words like "amount" matched fee text. Fixed by tuning `MIN_SCORE` and adding the
"not found" rule in the prompt.

## Limitations
Scanned PDFs need OCR; gemma:2b can miss subtle conflicts; threshold needs tuning per document set.
