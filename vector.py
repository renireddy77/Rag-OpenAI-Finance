# importing the libraries
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document # this helps create document objects
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv() # this reads .env file

api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(api_key=api_key)

# defining the location where vector db will be stored
db_location = "./chroma_financial_db1"
# Load the CSV file
try:
    df = pd.read_csv("Financials.csv")
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: str(x).strip().strip('"') if pd.notna(x) else x)
except FileNotFoundError:
    print("Error: Financials.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Check if the database already exists
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    # Iterate row by row in the dataframe to create documents
    for i, row in df.iterrows():
        # Combine relevant financial information into page_content
        # You can customize which columns are most important for your queries
        page_content = (
    f"In the {row['Segment']} segment, the product '{row['Product']}' was sold in {row['Country']}. "
    f"It generated gross sales of {row['Gross Sales']} and resulted in a profit of {row['Profit']}. "
    f"This transaction occurred on {row['Date']}."
)


        # Metadata: non-vectorized data stored alongside the vector.
        # This can be used for filtering later.
        metadata = {
            "segment": row.get('Segment', 'N/A'),
            "country": row.get('Country', 'N/A'),
            "year": pd.to_datetime(row['Date']).year if 'Date' in row and pd.notna(row['Date']) else 'N/A'
        }

        document = Document(
            page_content=page_content,
            metadata=metadata,
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

    # Create the vector store
    # This will persist the embeddings to the specified directory
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="Financial_Data", # Changed collection name
        persist_directory=db_location
    )
    print(f"Vector store created and persisted at {db_location}")
else:
    # Load the existing vector store if it already exists
    vector_store = Chroma(
        collection_name="Financial_Data", 
        persist_directory=db_location,
        embedding_function=embeddings
    )
    print(f"Vector store loaded from {db_location}")

# Treat the vector store as a retriever
# The retriever will look for relevant documents based on a query
retriever = vector_store.as_retriever(
    # search_kwargs (k parameter): tells the retriever to find and return "n" most relevant documents
    search_kwargs={"k": 5}
)

print("Retriever initialized successfully.")