import os
import json
import tempfile
from pdf2image import convert_from_path
import google.generativeai as genai
from dotenv import load_dotenv
from gemini_model_rotator import ModelManager
from google.api_core import exceptions
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter

# Configuration
input_directory = './input_files'   # Directory containing PDF files
output_folder = 'markdown_output'   # Folder to save Markdown files
chunk_size = 10                     # Number of pages per chunk
tracker_file = 'tracker.json'       # JSON file to track processed files

PROMPT = '''Examine the image and return all of the text within it, converted to
Markdown. Make sure the text reflects how a human being would read this,
following columns and understanding formatting. Ignore footnotes and
page numbers - they should not be returned as part of the Markdown.
Only generate markdown for the text found on the page. Every page top contains the School and Department name.
Use it as part of markdown to determine which school and department the text belongs to.'''

# Load environment variables before using them
load_dotenv()
genai.configure(api_key=os.getenv('GENAI_API_KEY'))

def load_tracker():
    """Load the tracker JSON file or create a new one if it doesn't exist."""
    if os.path.exists(tracker_file):
        with open(tracker_file, 'r') as f:
            return json.load(f)
    else:
        return {"tracked": []}

def save_tracker(tracker_data):
    """Save the updated tracker data to the JSON file."""
    with open(tracker_file, 'w') as f:
        json.dump(tracker_data, f, indent=2)

def get_unprocessed_files(input_dir, tracker_data):
    """Get a list of PDF files in the input directory that haven't been processed yet."""
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    tracked_files = tracker_data.get("tracked", [])
    unprocessed_files = [f for f in all_files if f not in tracked_files]
    return unprocessed_files

def split_pdf_into_chunks(pdf_path, chunk_size):
    """Split a large PDF into smaller chunks and return paths to the chunk files."""
    pdf = PdfReader(pdf_path)
    total_pages = len(pdf.pages)
    chunk_paths = []

    # Create a temporary directory to store PDF chunks
    temp_dir = tempfile.mkdtemp()

    for i in range(0, total_pages, chunk_size):
        # Create a new PDF writer
        pdf_writer = PdfWriter()

        # Calculate end page for this chunk
        end_page = min(i + chunk_size, total_pages)

        # Add pages to the writer
        for page_num in range(i, end_page):
            pdf_writer.add_page(pdf.pages[page_num])

        # Save the chunk
        chunk_path = os.path.join(temp_dir, f"chunk_{i//chunk_size + 1}.pdf")
        with open(chunk_path, 'wb') as output_file:
            pdf_writer.write(output_file)

        chunk_paths.append((chunk_path, i))  # Store path and starting page number

    return chunk_paths, temp_dir, total_pages

def gemini_model_processor(image, prompt, gemini_model):
    """Process a PIL image with the Gemini model to generate Markdown."""
    if gemini_model is None:
        raise ValueError("No available models to process the request.")
    print(f"Using model: {gemini_model.name}")

    generation_config = {
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": gemini_model.top_k,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name=gemini_model.name,
        generation_config=generation_config,
    )

    # Convert PIL image to BytesIO for uploading
    image_bytes = BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    # Upload the image to Gemini API
    uploaded_file = genai.upload_file(image_bytes, mime_type="image/jpeg")

    # Generate Markdown from image and prompt
    response = model.generate_content([uploaded_file, prompt])
    gemini_model.increment_usage()

    return response

def process_chunk(chunk_path, start_page, model_manager, output_folder, filename, poppler_path=None):
    """Process a single PDF chunk."""
    # Get an available model
    gemini_model = model_manager.get_available_model()

    # Convert PDF chunk to images
    print(f"Converting chunk starting at page {start_page + 1} to images...")
    images = convert_from_path(chunk_path, dpi=400, poppler_path=poppler_path)
    print(f"Converted {len(images)} pages to images.")

    # Get base filename without extension for use in the output filename
    base_filename = os.path.splitext(filename)[0]

    # Process each image in the chunk
    for i, image in enumerate(images, start=1):
        actual_page = start_page + i
        print(f"Processing page {actual_page} of {filename}")

        while True:
            if gemini_model is None:
                print("No available models. Skipping page.")
                break

            try:
                response = gemini_model_processor(image, PROMPT, gemini_model)
                markdown_text = response.text

                # Create output directory if it doesn't exist
                os.makedirs(output_folder, exist_ok=True)

                # Save the Markdown to a file with filename included for easy identification
                output_file = os.path.join(output_folder, f'{base_filename}_page_{actual_page}.md')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_text)
                print(f"Saved Markdown for page {actual_page} to {output_file}")
                break  # Success, move to next page

            except exceptions.ResourceExhausted:
                print(f"Resource exhausted for model {gemini_model.name}. Switching model.")
                gemini_model = model_manager.swap_model(gemini_model.name)

            except Exception as e:
                print(f"Error processing page {actual_page}: {e}")
                break  # Skip page on other errors

def process_pdf(pdf_path, filename, model_manager, output_folder, poppler_path=None):
    """Process a single PDF file."""
    print(f"\nProcessing file: {filename}")

    # Step 1: Split the PDF into smaller chunks
    print(f"Splitting PDF into chunks of {chunk_size} pages...")
    chunk_paths, temp_dir, total_pages = split_pdf_into_chunks(pdf_path, chunk_size)
    print(f"Split PDF into {len(chunk_paths)} chunks. Total pages: {total_pages}")

    # Step 2: Process each chunk
    for chunk_path, start_page in chunk_paths:
        process_chunk(chunk_path, start_page, model_manager, output_folder, filename, poppler_path)

    # Clean up the temporary directory
    import shutil
    shutil.rmtree(temp_dir)

    print(f"Finished processing {filename}")
    return True

def main():
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the tracker data
    tracker_data = load_tracker()

    # Get unprocessed files
    unprocessed_files = get_unprocessed_files(input_directory, tracker_data)

    if not unprocessed_files:
        print("No new files to process. All files have been already processed.")
        return

    print(f"Found {len(unprocessed_files)} new files to process: {', '.join(unprocessed_files)}")

    # Initialize the Gemini model manager
    model_manager = ModelManager('models.json')

    # Set poppler path if needed (for Windows)
    poppler_path = 'C:\\Program Files\\poppler-24.08.0\\Library\\bin'

    # Process each unprocessed file
    for filename in unprocessed_files:
        pdf_path = os.path.join(input_directory, filename)

        # Process the PDF
        success = process_pdf(pdf_path, filename, model_manager, output_folder, poppler_path)

        # Update the tracker if processing was successful
        if success:
            tracker_data["tracked"].append(filename)
            save_tracker(tracker_data)
            print(f"Added {filename} to tracker")

    print("\nBatch processing complete! All new files have been processed.")

if __name__ == "__main__":
    main()