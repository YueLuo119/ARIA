# ------------------ 调试开关 ------------------
DEBUG = False  # True 可输出内部日志

import os
from langchain_community.vectorstores import FAISS
from llama_index.readers.docling import DoclingReader

from langchain_core.documents import Document  # Optional if you need LangChain docs
from pathlib import Path
from langchain_openai import OpenAIEmbeddings#

from typing import TypedDict, List, Optional, Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

import streamlit as st

st.title('ARIA Web App')
st.divider()#adds a horizontal rule (a visual dividing line)
st.write('Enter your question related to the course:')
st.divider()

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# 1. API key
def _set_env(key: str):
    os.environ['OPENAI_API_KEY'] = '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'

_set_env("OPENAI_API_KEY")

from langchain.chat_models import init_chat_model
llm = init_chat_model("openai:gpt-4.1", temperature=0)

# 2. read the pdf
# exist pdf
import os
project_root = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = project_root
pdf_library = {
    "chapter1": os.path.join(data_dir, "chapter1.docx"),
    "chapter2": os.path.join(data_dir, "chapter2.docx"),
    "chapter3": os.path.join(data_dir, "chapter3.docx"),
    "course information": os.path.join(data_dir, "course syllabus.docx")
}
# safety check
for name, path in pdf_library.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found: '{path}' for document '{name}'")
@tool
def greeting_tool(query: str):
    """Use this tool to respond to simple user greetings like 'hi', 'hello', or 'hey'.
       Use it only when the user is not asking a question.
    """
    return "My name is Aria, what can I help you?"

tools = [greeting_tool]
llm_with_tools = llm.bind_tools(tools)

class GraphState(TypedDict):
    question: str
    context: Optional[str]
    messages: List[BaseMessage]
    selected_file: Optional[str]

def tool_check(state: GraphState) -> dict:
    """
    deciding whether to call a tool or continue.
    """
    messages = state['messages']
    # The agent decides whether to call a tool based on the conversation history.
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def select_pdf_from_question(state: GraphState) -> dict:
    """Selects the most relevant PDF based on the user's question."""
    question = state["question"]
    pdf_prompt = (
        "You are an AI assistant tasked with selecting the most relevant document "
        "from the following list based on the user's question.\n\n"
        f"Available documents:\n{chr(10).join(list(pdf_library.keys()))}\n\n"
        f"User's question:\n{question}\n\n"
        "Which document best matches the user's question? "
        "Respond with the exact document name (not filename)."
        "if reselect the document from rewrite_question, please choose another document to answer."
    )
    response = llm.invoke(pdf_prompt)
    selected_name = response.content.strip().lower()

    # Find the corresponding file path from the library
    selected_file = pdf_library.get(selected_name)


    # If still no match, default to the first PDF to avoid errors
    if not selected_file:
        selected_name = list(pdf_library.keys())[0]
        selected_file = pdf_library[selected_name]
        print(f"Warning: No clear document match found. Defaulting to '{selected_file}'")

    if DEBUG:
        print(f"Selected PDF: {selected_file}")

    # Return a dictionary to update the graph's state
    return {
        "selected_file": selected_file
    }


persist_directory = str(Path.home() / "aria_vectorstore")
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# 3.retriever_tool
###
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState

# 5.1 conditional_edge
def retrieve_and_answer_node(state: GraphState) -> dict:
    """Generate answer with optional retrieval (internal helper)."""
    global retriever, retriever_tool
    question = state["question"]
    selected_file = state["selected_file"]

    if not selected_file:
        raise ValueError("Error: 'selected_file' is not set in the state.")


    SOURCE = str(Path(selected_file).resolve())
    reader = DoclingReader()
    documents = reader.load_data(SOURCE)
    doc_splits = [Document(page_content=node.text, metadata=node.metadata) for node in documents]

    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(doc_splits, embedding)
    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    retrieved_docs = retriever.invoke(question)
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    # Create retriever tool
    system_prompt = (
        "You are a helpful assistant. Use the retrieved context below to answer the user's question. "
        "If you are uncertain, simply state you do not know. "
        "Respond in English and, if the answer contains mathematical formulas, wrap inline TeX with `$...$` and block formulas with `$$...$$` so they render correctly.\n\n"
        "CONTEXT:\n{context}\n\n"
        "USER QUESTION: {question}"
    )
    generation_prompt = system_prompt.format(context=context, question=question)

    response = llm.invoke([HumanMessage(content=generation_prompt)])

    # Append the AI's response to the message list
    return {"context": [response]}
#test
#input = {"messages": [{"role": "user", "content": "What's AAPL?"}]}
#generate_query_or_respond(input)["messages"][-1].pretty_print()
#tools = [retriever_tool]
#tools_dict = {our_tool.name: our_tool for our_tool in tools}

# 5.2 score the answer
from pydantic import BaseModel, Field
from typing import Literal



class GradeDocuments(BaseModel):
    """Binary score for relevance check."""
    binary_score: Literal['yes', 'no'] = Field(description="Relevance score: 'yes' if relevant, 'no' if not.")

def grade_documents(state: GraphState) -> Literal["generate_answer", "rewrite_question"]:
    # already input MessageState
    # Literal is a hint that helps tools or LLM agents expect only those two strings return options.

    """Determine whether the retrieved documents are relevant to the question."""
    question = state["question"]
    context = state["context"]

    GRADE_PROMPT = (
        "You are a grader assessing relevance of a retrieved document to a user question. \n "
        "Here is the retrieved document: \n\n {context} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )


    grading_prompt = GRADE_PROMPT.format(question=question, context=context)

    # Use structured output for a reliable 'yes' or 'no'
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    response = structured_llm_grader.invoke(grading_prompt)

    if DEBUG:
        print(f"Relevance Grade: {response.binary_score}")
    if response.binary_score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

# 5.3 build the rewrite_question node.
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)
def rewrite_question(state: GraphState) -> dict:
    """Rewrite the original user question."""
    question = state["question"]
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([{"role": "user", "content": prompt}])
    rewritten_question = response.content
    return {"question": rewritten_question}

# 5.4 build generate_answer node
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the retrieved context to answer the question. If unsure, say you don't know. "
    "Respond in English. When your answer contains mathematical expressions, format them with TeX wrapped in `$...$` (inline) or `$$...$$` (block) so that they render on the client side. "
    "Limit the answer to three concise sentences.\n"
    "Question: {question} \n"
    "Context: {context}"
)
def generate_answer(state: GraphState) -> dict:
    """Generate an answer."""
    question = state["question"]
    context = state["context"]
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

from langgraph.graph import StateGraph, START, END#
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

graph = StateGraph(GraphState)

graph.add_node('tool_check',tool_check)
graph.add_node("tools", ToolNode(tools))
graph.add_node("retrieve_and_answer_node", retrieve_and_answer_node)
graph.add_node("select_pdf", select_pdf_from_question)
#graph.add_node("retrieve", ToolNode([retriever_tool]))
graph.add_node(rewrite_question)
graph.add_node(generate_answer)

graph.add_edge(START, "tool_check")
graph.add_conditional_edges(
    "tool_check",
    tools_condition,
    {
        "tools": 'tools',
        END: "select_pdf",  # If no tools are called, continue to PDF selection
    }
)
# If a tool is used, the result is printed and the graph ends for this example.
# You could connect this to another node to process the tool's output.
graph.add_edge("tools", END)

graph.add_edge("select_pdf", "retrieve_and_answer_node")

graph.add_conditional_edges(
    "retrieve_and_answer_node",
    # Assess agent decision
    grade_documents,
)
graph.add_edge("generate_answer", END)
graph.add_edge("rewrite_question", "select_pdf")

rag_agent = graph.compile()

# 6.run the agentic RAG

def submit_question():
    """Streamlit 回调：处理用户输入并清空输入框"""
    q = st.session_state.get("user_question", "").strip()
    if not q:
        return
    initial_state = {
        "question": q,
        "messages": [HumanMessage(content=q)]
    }
    try:
        final_state = rag_agent.invoke(initial_state)
        ans = final_state["messages"][-1].content
        st.session_state.history.append((q, ans))
    except Exception as e:
        st.session_state.history.append((q, f"❌ 发生错误: {e}"))
    # 清空输入框
    st.session_state.user_question = ""


def running_agent():
    if "history" not in st.session_state:
        st.session_state.history = []

    st.text_input(
        "Please enter your question related to the course:",
        key="user_question",
        on_change=submit_question,
    )

    st.divider()
    for q, a in st.session_state.history[::-1]:
        st.markdown(f"**User：** {q}")
        st.markdown(f"**ARIA：** {a}", unsafe_allow_html=True)
        st.divider()

# Create dummy docx files for testing if they don't exist
for doc in pdf_library.values():
    if not os.path.exists(doc):
        with open(doc, 'w') as f:
            f.write(f"This is a dummy file named {doc}.")

running_agent()

try:
    image_data = rag_agent.get_graph().draw_mermaid_png()
    with open("agent_select_multi_tool_check.png", "wb") as f:
        f.write(image_data)
    print("Graph image saved as agent_version1_select_multi.png")
except Exception as e:
    print(f"Could not generate graph image: {e}")
