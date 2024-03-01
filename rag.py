import os
import openai
import langchain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from data_ingestion import QdrantIndexer
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]
qdrant_config = QdrantIndexer()
langchain.debug = True


class RAG:
    def __init__(self,
                 chat_max_tokens: int = 3097,
                 model_name="gpt-3.5-turbo",
                 creativeness: float = 0.7):
        self.vector_store = qdrant_config.get_vector_store()
        self.model = self.set_llm_model(model_name, creativeness)
        self.retriever = self.set_retriever()
        self.chat_history = self.set_chat_history(max_token_limit=chat_max_tokens)

    def set_llm_model(self, model_name="gpt-3.5-turbo", temperature: float = 0.7):
        return ChatOpenAI(model_name=model_name, temperature=temperature)

    def set_retriever(self):
        # Self-Querying Retriever
        jobs_metadata_info = [
            AttributeInfo(
                name="company_name",
                description="The name of the company.",
                type="string"
            ),
            AttributeInfo(
                name="city",
                description="The city where the job position is located.",
                type="string"
            ),
            AttributeInfo(
                name="region",
                description="The region where the job position is located.",
                type="string"
            ),
            AttributeInfo(
                name="country",
                description="The country where the job position is located.",
                type="string"
            ),
            AttributeInfo(
                name="post_year",
                description="The year when the job was posted.",
                type="integer",
            ),
            AttributeInfo(
                name="post_month",
                description="The month of the year when the job was posted.",
                type="integer",
            ),
            AttributeInfo(
                name="post_day",
                description="The day of the month when the job was posted.",
                type="integer",
            ),
            AttributeInfo(
                name="job_position",
                description="The title of the posted job.",
                type="string",
            )
        ]
        document_content_description = "A description of the job position"

        retriever = SelfQueryRetriever.from_llm(
            self.model,
            self.vector_store,
            document_content_description,
            jobs_metadata_info,
        )
        return retriever

    def set_chat_history(self, max_token_limit):
        return ConversationTokenBufferMemory(llm=self.model, max_token_limit=max_token_limit, return_messages=True)

    def ask(self, question: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a bot that answers user questions based on the context provided. If you don’t know the answer, "
             "just say that you don’t know, don’t try to make up an answer.Context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        output_parser = StrOutputParser()
        chain = prompt | self.model | output_parser
        answer = chain.invoke({
            "input": question,
            "chat_history": self.chat_history.load_memory_variables({})['history'],
            "context": self.retriever.get_relevant_documents(question)
        })

        self.chat_history.save_context({"input": question}, {"output": answer})
        return answer


rag = RAG()
while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break
    response = rag.ask(user_input)
    print("Bot:", response)
