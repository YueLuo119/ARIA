import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from llama_index.readers.docling import DoclingReader
from langchain_core.documents import Document
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from typing import TypedDict, List, Optional, Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition


# --- Page Configuration (Set this at the very top) ---
st.set_page_config(layout="wide")

# --- Define the three main columns for the entire page ---
#col1, col2, col3 = st.columns([1, 1, 1])
col_left, col1, col2, col3, col_right = st.columns([0.6,0.1,3,0.3,1])

# --- Content for the Left Column (col1) ---
with col_left:
    #st.sidebar.title("Dashboard Menu")

    # Using st.expander for the "Registration" section
    with st.sidebar.expander("Contact"):
        st.write("Professor: Somdatta Goswami")
        st.write("somdatta@jhu.edu")
        st.write("TA: Dibakar Roy Sarkar")
        st.write("droysar1@jh.edu")
 

    # You can add other top-level items or other expanders
    with st.sidebar.expander("Classtime"):
        st.write("Update My Information")
       
    with st.sidebar.expander("Reference Book"):
        st.write("Update My Information")
    

    # A disabled menu item for visual representation
    st.sidebar.markdown("---")


# --- Content for the Middle Column (col2) ---
with col2:
    # ------------------ 调试开关 ------------------
    DEBUG = False  # True 可输出内部日志

    st.title("Hi, I'm Aria🤗Your AI assistant for the course!")
    st.divider()#adds a horizontal rule (a visual dividing line)
    st.write("Before you ask me a question, please read 'How to write a prompt' and select the chapter you want to ask me about.")
    st.write("Any other questions which cannot be solved, please contact TA or Professor.")
    st.divider()

    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


  
    llm = init_chat_model("openai:gpt-4.1", temperature=0)

    # 2. read the pdf

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = project_root # Assuming docx files are in the parent directory of the script's directory
    pdf_library = {
        "chapter1": os.path.join(data_dir, "chapter1.docx"),
        "chapter2": os.path.join(data_dir, "chapter2.docx"),
        "chapter3": os.path.join(data_dir, "chapter3.docx"),
        "course information": os.path.join(data_dir, "course syllabus.docx")
    }
    # safety check
    for name, path in pdf_library.items():
        if not os.path.exists(path):
            st.error(f"File not found: '{path}' for document '{name}'")
            # Create dummy files if they don't exist, for testing purposes
            with open(path, 'w') as f:
                f.write(f"This is a dummy file for {name}.")


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
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def select_pdf_from_question(state: GraphState) -> dict:
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
        selected_file = pdf_library.get(selected_name)

        if not selected_file:
            selected_name = list(pdf_library.keys())[0]
            selected_file = pdf_library[selected_name]
            if DEBUG:
                print(f"Warning: No clear document match found. Defaulting to '{selected_file}'")

        return {"selected_file": selected_file}

    persist_directory = str(Path.home() / "aria_vectorstore")
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    def retrieve_and_answer_node(state: GraphState) -> dict:
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

        system_prompt = (
            "You are a helpful assistant. Use the retrieved context below to answer the user's question. "
            "If you are uncertain, simply state you do not know. "
            "Respond in English and, if the answer contains mathematical formulas, wrap inline TeX with `$...$` and block formulas with `$$...$$` so they render correctly.\n\n"
            "CONTEXT:\n{context}\n\n"
            "USER QUESTION: {question}"
        )
        generation_prompt = system_prompt.format(context=context, question=question)
        response = llm.invoke([HumanMessage(content=generation_prompt)])
        return {"context": [response]}


    class GradeDocuments(BaseModel):
        binary_score: Literal['yes', 'no'] = Field(description="Relevance score: 'yes' if relevant, 'no' if not.")

    def grade_documents(state: GraphState) -> Literal["generate_answer", "rewrite_question"]:
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
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        response = structured_llm_grader.invoke(grading_prompt)

        if DEBUG:
            print(f"Relevance Grade: {response.binary_score}")
        return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

    REWRITE_PROMPT = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:"
        "\n ------- \n"
        "{question}"
        "\n ------- \n"
        "Formulate an improved question:"
    )
    def rewrite_question(state: GraphState) -> dict:
        question = state["question"]
        prompt = REWRITE_PROMPT.format(question=question)
        response = llm.invoke([{"role": "user", "content": prompt}])
        return {"question": response.content}

    GENERATE_PROMPT = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer the question. If unsure, say you don't know. "
        "Respond in English. When your answer contains mathematical expressions, format them with TeX wrapped in `$...$` (inline) or `$$...$$` (block) so that they render on the client side. "
        "Limit the answer to three concise sentences.\n"
        "Question: {question} \n"
        "Context: {context}"
    )
    def generate_answer(state: GraphState) -> dict:
        question = state["question"]
        context = state["context"]
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = llm.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}


    graph = StateGraph(GraphState)
    graph.add_node('tool_check', tool_check)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("retrieve_and_answer_node", retrieve_and_answer_node)
    graph.add_node("select_pdf", select_pdf_from_question)
    graph.add_node(rewrite_question)
    graph.add_node(generate_answer)
    graph.add_edge(START, "tool_check")
    graph.add_conditional_edges("tool_check", tools_condition, {"tools": 'tools', END: "select_pdf"})
    graph.add_edge("tools", END)
    graph.add_edge("select_pdf", "retrieve_and_answer_node")
    graph.add_conditional_edges("retrieve_and_answer_node", grade_documents)
    graph.add_edge("generate_answer", END)
    graph.add_edge("rewrite_question", "select_pdf")
    rag_agent = graph.compile()

    def submit_question():
        q = st.session_state.get("user_question", "").strip()
        if not q:
            return
        initial_state = {"question": q, "messages": [HumanMessage(content=q)]}
        try:
            final_state = rag_agent.invoke(initial_state)
            ans = final_state["messages"][-1].content
            st.session_state.history.append((q, ans))
        except Exception as e:
            st.session_state.history.append((q, f"❌ An error occurred: {e}"))
        st.session_state.user_question = ""

    if "history" not in st.session_state:
        st.session_state.history = []

    st.text_input(
        "Please enter your question related to the course:",
        key="user_question",
        on_change=submit_question,

    )

    st.divider()
    for q, a in st.session_state.history[::-1]:
        st.markdown(f"**User:** {q}")
        st.markdown(f"**ARIA:** {a}", unsafe_allow_html=True)
        st.divider()


# --- Content for the Right Column (col3) ---

with col_right:
    st.markdown("""
    <style>
    .custom-font {
        font-family: 'Georgia', serif;
        font-size: 17px;
        color: #2e2e2e;
        line-height: 1.6;
    }
    .custom-font h3 {
        margin-top: 1em;
        font-size: 22px;
        color: #111;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='custom-font'>", unsafe_allow_html=True)

    st.markdown("### Catalogue", unsafe_allow_html=True)
    st.markdown("Please choose the chapter you want to ask about and write it in the front of the question:", unsafe_allow_html=True)
    st.markdown("For instance: Use chapter2, what's plane stress?", unsafe_allow_html=True)
    st.divider()

    st.markdown("#### Chapter1", unsafe_allow_html=True)
    st.markdown("""
    • Common causes of Failure in Machines and Structures  
    • Histories – Fracture mechanics  
    • What is fracture mechanics  
    • Approaches used in fracture mechanics  
    • Definition of Brittle fracture and ductile fracture
    """, unsafe_allow_html=True)

    st.markdown("#### Chapter2", unsafe_allow_html=True)
    st.markdown("""
    • Displacements and strains  
    • Stresses  
    • Stress transformation  
    • Principal Stress  
    • Equilibrium equations  
    • Constitutive equations  
    • Compatibility conditions  
    • Plane stress state and plane strain state  
    • Total potential energy
    """, unsafe_allow_html=True)

    st.markdown("#### Chapter3", unsafe_allow_html=True)
    st.markdown("""
    • Griffith Postulation  
    • Energy release rate (Irwin, 1956)  
    • Energy Release Rate in the Forms of Compliance change  
    • G in terms of Strain Energy Change  
    • Energy Release Rate of DCB specimen  
    • Inelastic Deformation Effect at Crack Tip  
    • Crack Resistance and Instability  
    • The Effect of Thickness of the Specimen
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    #selected_chapter = st.radio("Chapters", list(pdf_library.keys()))

    #st.write(f"You have selected: **{selected_chapter}**")
