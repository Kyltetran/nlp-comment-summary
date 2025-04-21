
import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from get_embedding_function import get_embedding_function
import os

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question below based on the context provided:
{context}
---
You are a YouTube comment summarizer. You can summarize the comments provided with insightful and informative analysis. You can also answer the questions relating to that video comments: {question}.
"""


def query_rag(query_text: str):
    # Prepare the DB
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())

    # Search the DB
    results = db.similarity_search_with_score(
        query_text, k=2000)  # Increase k to get more results (more text)

    # Prepare the context text from the search results
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

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
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)
