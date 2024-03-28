from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.base import Chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def query_document(
    docs: list[Document],
    user_query: str,
    model_name: str,
    openai_api_key: str,
    base_url: str,
    temperature: float = 0.3,
) -> str:
    pass

    # Define LLM chain
    llm = ChatOpenAI(
        temperature=temperature,
        model_name=model_name,
        api_key=openai_api_key,
        base_url=base_url,
    )
    chain = get_map_reduce_chain(llm, user_query=user_query)

    result = chain.invoke(docs)
    return result["output_text"]


def get_map_reduce_chain(llm: ChatOpenAI, user_query: str) -> Chain:
    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of documents, please identify the information that is most relevant to the following query:
    {user_query} 
    If the document is not relevant, please write "not relevant".
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_prompt = map_prompt.partial(user_query=user_query)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    # Reduce
    reduce_template = """The following is set of partial answers to a user query:
    {docs}
    Take these and distill it into a final, consolidated answer to the following query:
    {user_query} 
    Complete Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_prompt = reduce_prompt.partial(user_query=user_query)

    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )
    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    return map_reduce_chain
