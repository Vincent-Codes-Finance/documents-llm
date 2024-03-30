# documents-llm

Sample cripts to summarize and query documents using LLMs.

THIS IS NOT ROBUST OR PRODUCTION-READY! FOR DEMONSTRATION PURPOSES ONLY!

## References

This is the sample code for the following video and blog post:

- [Summarize and Query PDFs with a Private Local GPT for Free using Ollama and Langchain (YouTube)](https://youtu.be/Tnu_ykn1HmI)
- [Summarize and Query PDFs with AI using Ollama](https://vincent.codes.finance/posts/documents-llm/)

## Setup

### Dependencies

All you have to do is install the dependencies in `pyproject.toml`:
```
python = "^3.12"
openai = "^1.14.3"
langchain = "^0.1.13"
ollama = "^0.1.8"
rich = "^13.7.1"
python-dotenv = "^1.0.1"
langchain-openai = "^0.1.1"
pypdf = "^4.1.0"
tiktoken = "^0.6.0"
```

Using poetry, that would be:

```bash
poetry install
```

and setup your environment variables. The recommended way is to use a `.env` file. Just copy
and rename one of `.env-ollama-sample` or `.env-openai-sample` to `.env`. If you use
OpenAI, you will need to also set your API key in `.env`


## Streamlit App Usage

```bash
streamlit run doc_app.py
```

## CLI Usage

There are two scripts, one for summarizing and one for querying documents.

### Summary

To summarize `document.pdf` from the first page, excluding the last two, using mixtral with a temperature of 0.2:

```bash
python summarize.py document.pdf -s 0 -e "-2" -m mixtral -t 0.2
```

### Query

To query `document.pdf` from the first page, excluding the last two, using mixtral with a temperature of 0.2:

```bash
python query.py document.pdf "What is the data used in this paper?" -s 0 -e "-2" -m mixtral -t 0.2
```