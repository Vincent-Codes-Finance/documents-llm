from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def summarize_document(
    docs: list[Document],
    model_name: str,
    openai_api_key: str,
    base_url: str,
    temperature: float = 0.1,
) -> str:
    pass

    # Define LLM chain
    llm = ChatOpenAI(
        temperature=temperature,
        model_name=model_name,
        api_key=openai_api_key,
        base_url=base_url,
    )

    prompt_template = """Write a long summary of the following document. 
    Only include information that is part of the document. 
    Do not include your own opinion or analysis.
    
    Document:
    "{document}"
    Summary:"""
    prompt = PromptTemplate.from_template(prompt_template)
    prompt = prompt.partial()

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="document"
    )
    result = stuff_chain.invoke(docs)
    return result["output_text"]


# def get_map_reduce_chain(llm: ChatOpenAI) -> Chain:
#     # Map
#     map_template = """The following is a set of documents
#     {docs}
#     Based on this list of docs, please identify the main themes
#     Helpful Answer:"""
#     map_prompt = PromptTemplate.from_template(map_template)
#     map_chain = LLMChain(llm=llm, prompt=map_prompt)
#     # Reduce
#     reduce_template = """The following is set of summaries:
#     {docs}
#     Take these and distill it into a final, consolidated summary of the main themes.
#     Helpful Answer:"""
#     reduce_prompt = PromptTemplate.from_template(reduce_template)

#     # Run chain
#     reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

#     # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
#     combine_documents_chain = StuffDocumentsChain(
#         llm_chain=reduce_chain, document_variable_name="docs"
#     )

#     # Combines and iteratively reduces the mapped documents
#     reduce_documents_chain = ReduceDocumentsChain(
#         # This is final chain that is called.
#         combine_documents_chain=combine_documents_chain,
#         # If documents exceed context for `StuffDocumentsChain`
#         collapse_documents_chain=combine_documents_chain,
#         # The maximum number of tokens to group documents into.
#         token_max=4000,
#     )
#     # Combining documents by mapping a chain over them, then combining results
#     map_reduce_chain = MapReduceDocumentsChain(
#         # Map chain
#         llm_chain=map_chain,
#         # Reduce chain
#         reduce_documents_chain=reduce_documents_chain,
#         # The variable name in the llm_chain to put the documents in
#         document_variable_name="docs",
#         # Return the results of the map steps in the output
#         return_intermediate_steps=False,
#     )

#     return map_reduce_chain
