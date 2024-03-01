import pandas as pd
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DataFrameLoader


class QdrantIndexer:
    def __init__(self, model_name="BAAI/bge-large-en", collection_name="job_db", qdrant_url="http://localhost:8080"):
        self.model_name = model_name
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={"normalize_embeddings": False}
        )

    def load_data(self, data_path, page_content_column):
        try:
            jobs_df = pd.read_csv(data_path)
            loader = DataFrameLoader(jobs_df, page_content_column=page_content_column)
            return loader.load()
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{data_path}' not found.")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def create_qdrant_index(self, data_path, page_content_column):
        try:
            documents = self.load_data(data_path, page_content_column)
            qdrant = Qdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                url=self.qdrant_url,
                collection_name=self.collection_name
            )
            return qdrant
        except Exception as e:
            raise Exception(f"Error creating Qdrant index: {str(e)}")

    def get_vector_store(self):
        try:
            client = QdrantClient(
                url=self.qdrant_url
            )
            return Qdrant(
                client=client,
                embeddings=self.embeddings,
                collection_name=self.collection_name
            )
        except Exception as e:
            raise Exception(f"Error getting vector store: {str(e)}")


# Example usage:
data_path = "data/jobs_cleaned.csv"
page_content_column = "clean_job_description"

qdrant_indexer = QdrantIndexer()
qdrant_indexer.create_qdrant_index(data_path, page_content_column)
