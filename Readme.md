# RAG Application Using Gemini Model

## Overview
This is a Retrieval-Augmented Generation (RAG) application that leverages Google's Gemini model as the backend. The project is managed using `uv` for dependency management and includes three main Python scripts.

## Prerequisites
- Install `uv` on your system.
- Obtain a Gemini API key and store it in a `.env` file with the following format:
  ```
  GENAI_API_KEY=your_api_key_here
  ```

## Installation
1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd <project_directory>
   ```
2. Initialize the project using `uv`:
   ```sh
   uv init
   ```
3. Install dependencies:
   ```sh
   uv sync
   ```

## Usage
1. **Convert PDFs to Markdown**
   Place the PDF files that need to be converted in the `input_files` directory, then run:
   ```sh
   uv run pdftomd.py
   ```
   The converted Markdown files will be stored in the `markdown_output` folder.

2. **Chat with the Processed PDFs**
   Once the Markdown files are available, you can start interacting with the content using:
   ```sh
   uv run chat.py
   ```

3. **Model Rotation to Avoid Rate Limits**
   The `gemini_model_rotator.py` script ensures seamless API usage by rotating the model as needed to prevent hitting the 429 rate limit. No need to run this script manually; it is called internally by the `chat.py` script. But need to pass a models.json file as provided above to run with those particular models.

## Dependencies
This project uses libraries such as:
- `langchain`
- `bs4` (BeautifulSoup)
- `pdf2image`
- Other dependencies listed in the `pyproject.toml` file

## Notes
- Ensure that the `.env` file is correctly set up before running the scripts.
- Use `uv` to run scripts and manage dependencies.

---

This README provides a clear guide to setting up and running the project. Let me know if you need any refinements!
