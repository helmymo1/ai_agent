import os
import numpy as np
from app import helper

def run_test():
    """
    A simple script to test the core application logic without running a server.
    """
    # 1. Define the test file and query
    pdf_path = "declaration_gutenberg.pdf"
    query = "When did the colonies declare their independence?"

    print(f"--- Starting Test ---")
    print(f"PDF: {pdf_path}")
    print(f"Query: '{query}'")

    # 2. Check if the PDF exists
    if not os.path.exists(pdf_path):
        print(f"Error: Test PDF not found at '{pdf_path}'")
        return

    try:
        # 3. Process the PDF
        print("\n--- Processing PDF ---")
        # Extract text
        text = helper.extract_text_from_pdf(pdf_path)
        print("Text extracted successfully.")
        print(f"Extracted text length: {len(text)}")
        print(f"Extracted text (first 500 chars): {text[:500]}")
        # Chunk text
        chunks = helper.chunk_text(text)
        print(f"{len(chunks)} chunks created.")
        # Generate embeddings
        model = helper.get_model()
        print("Model loaded.")
        embeddings = model.encode(chunks, show_progress_bar=True)
        print("Embeddings generated.")

        # 4. Perform search
        print("\n--- Performing Search ---")
        results = helper.search(query, chunks, embeddings, top_k=3)

        # 5. Display results
        print("\n--- Search Results ---")
        if not results:
            print("No results found.")
        else:
            for i, result in enumerate(results):
                print(f"Result {i+1} (Similarity: {result['similarity']:.4f}):")
                print(result['chunk'])
                print("-" * 25)

    except Exception as e:
        print(f"\nAn error occurred during the test: {str(e)}")

    print("\n--- Test Finished ---")

if __name__ == '__main__':
    run_test()
