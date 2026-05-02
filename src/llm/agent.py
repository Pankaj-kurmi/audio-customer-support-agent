import logging
from typing import Optional
import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseRetriever
import os

logger = logging.getLogger(__name__)

class CustomerSupportAgent:
    """
    AI agent for customer support with RAG (Retrieval-Augmented Generation).
    
    Uses ChromaDB for document retrieval and OpenAI for text generation.
    """

    def __init__(
        self,
        chroma_collection,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """
        Initialize the Customer Support Agent.

        Args:
            chroma_collection: ChromaDB collection for RAG
            llm_model: LLM model name (e.g., 'gpt-3.5-turbo', 'gpt-4')
            temperature: LLM temperature for response generation
            max_tokens: Maximum tokens in response
        """
        self.collection = chroma_collection
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
        )

        # System prompt for customer support
        self.system_prompt = """You are a helpful customer support representative. 
        Use the provided context from our knowledge base to answer customer questions accurately and helpfully.
        If you don't have relevant information in the context, be honest about it and provide general guidance.
        Keep responses concise, friendly, and professional."""

        logger.info(
            f"CustomerSupportAgent initialized with model={{llm_model}}, "
            f"temperature={{temperature}}, max_tokens={{max_tokens}}"
        )

    async def _rag_search(self, query: str) -> str:
        """
        Search for relevant documents in ChromaDB using RAG.

        Args:
            query: User query to search for

        Returns:
            Formatted string containing top 3 relevant documents with metadata

        Raises:
            ValueError: If query is empty
            Exception: If ChromaDB search fails
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to _rag_search")
            return "No relevant documents found."

        try:
            # Run ChromaDB query in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, self._sync_rag_search, query
            )
            return results

        except Exception as e:
            logger.error(f"Error during RAG search for query '{{query}}': {{str(e)}}")
            return "Error retrieving knowledge base documents."

    def _sync_rag_search(self, query: str) -> str:
        """
        Synchronous RAG search (wrapper for async execution).

        Args:
            query: Search query

        Returns:
            Formatted context string
        """
        try:
            # Query ChromaDB collection
            results = self.collection.query(
                query_texts=[query],
                n_results=3,
            )

            if not results or not results.get("documents") or len(results["documents"]) == 0 or not results["documents"][0]:
                logger.info(f"No documents found for query: {{query}}")
                return "No relevant documents found in knowledge base."

            # Format results with metadata
            context_parts = []
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances), 1):
                if doc:  # Only include non-empty documents
                    title = metadata.get("title", "Unknown") if metadata else "Unknown"
                    relevance = 1 - distance if distance is not None else 0
                    
                    context_parts.append(
                        f"Document {{i}} - {{title}} (relevance: {{relevance:.2%}}):\n{{doc}}"
                    )

            context = "\n\n---\n\n".join(context_parts)
            logger.info(f"RAG search returned {{len(context_parts)}} documents")
            return context if context.strip() else "No relevant documents found."

        except Exception as e:
            logger.error(f"Error in _sync_rag_search: {{str(e)}}", exc_info=True)
            return "Error retrieving knowledge base documents."

    async def generate_response(self, query: str) -> str:
        """
        Generate AI response using RAG and LLM.

        Args:
            query: User query

        Returns:
            AI-generated response

        Raises:
            ValueError: If query is empty
            Exception: If LLM call fails
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to generate_response")
            return "Please provide a valid question."

        try:
            # Get RAG context
            logger.info(f"Searching RAG for query: {{query}}")
            rag_context = await self._rag_search(query)

            # Create prompt with context
            from langchain.prompts import PromptTemplate

            prompt_template = PromptTemplate(
                input_variables=["system_prompt", "rag_context", "query"],
                template="{{system_prompt}}\n\nKnowledge Base Context:\n{{rag_context}}\n\nCustomer Question:\n{{query}}\n\nResponse:",
            )

            prompt = prompt_template.format(
                system_prompt=self.system_prompt,
                rag_context=rag_context,
                query=query,
            )

            # Generate response (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self._sync_generate_response, prompt
            )

            logger.info(f"Generated response for query: {{query[:50]}}...")
            return response

        except Exception as e:
            logger.error(f"Error generating response for query '{{query}}': {{str(e)}}", exc_info=True)
            return "Sorry, I encountered an error processing your request. Please try again."

    def _sync_generate_response(self, prompt: str) -> str:
        """
        Synchronous response generation (wrapper for async execution).

        Args:
            prompt: Formatted prompt with context

        Returns:
            Generated response text
        """
        try:
            from langchain.schema import HumanMessage

            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            return response.content.strip()

        except Exception as e:
            logger.error(f"Error in LLM invocation: {{str(e)}}", exc_info=True)
            return "Error generating response from LLM."

    async def process_query(self, query: str) -> dict:
        """
        Process a customer query end-to-end.

        Args:
            query: Customer question

        Returns:
            Dictionary with response and metadata
        """
        try:
            response = await self.generate_response(query)
            return {
                "status": "success",
                "query": query,
                "response": response,
                "model": self.llm_model,
            }

        except Exception as e:
            logger.error(f"Error processing query: {{str(e)}}", exc_info=True)
            return {
                "status": "error",
                "query": query,
                "response": "Error processing your request.",
                "error": str(e),
            }