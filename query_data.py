
import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from get_embedding_function import get_embedding_function
import os

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a YouTube comment summarizer. Below is a collection of user comments extracted from a video.

{context}

---

Please write a summary highlighting the key points and general sentiment expressed in these comments.
Focus on providing a well-rounded overview in less than 5 paragraphs.
"""


def query_rag():
    # Load the Chroma vector store
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())

    # Retrieve relevant documents
    results = db.similarity_search_with_score(
        "summarize youtube comments", k=2000)

    # Build context string from retrieved documents
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])

    # Format prompt with context
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text)

    print("Prompt sent to model:")
    print(prompt)

    # Use OllamaLLM model without an API key
    model = OllamaLLM(model="llama3.2")  # Specify Ollama model here
    # Run the model and get the response
    response_text = model.invoke(prompt)  # Directly assign the string response

    # formatted_sources = [
    #     f"{doc.metadata.get('id', 'Unknown Source')}: {doc.page_content}" for doc, _score in results]

    # # Combine response with formatted sources
    # formatted_response = f"Response: {response_text}\nSources:\n" + \
    #     "\n\n".join(formatted_sources)

    # Save the output to a file instead of printing
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(response_text)

    print("Output saved to output.txt")


# Directly handle arguments without the need for a separate main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="summarize",
        help="Trigger the summarization task (default: summarize)"
    )
    args = parser.parse_args()

    if args.task == "summarize":
        query_rag()
