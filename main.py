from utils import process_pdf, find_most_similar, find_full_context

def main():
    # Load PDF once
    pdf_path = input("Give name of the PDF:")
    chunks, embeddings = process_pdf(pdf_path)
    
    # Interactive search loop
    while True:
        user_prompt = input("prompt: ")
        if user_prompt.lower() == 'exit':
            break
        
        # Find matches
        results = find_most_similar(user_prompt, chunks, embeddings, top_k=3)
        
        # Display results with context
        print(f"\nTop {len(results)} matches:")
        for idx, (chunk, score) in enumerate(results, 1):
            context = find_full_context(chunk, chunks)
            print(f"\nMatch #{idx} (Page {chunk['page']})")
            print(f"Context: {context}")
            print("-" * 80)

if __name__ == "__main__":
    main()