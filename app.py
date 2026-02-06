from __future__ import annotations

from pathlib import Path
import os
from typing import Optional, List

import streamlit as st
from dotenv import load_dotenv

# ========= LangChain / Providers =========
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings

# ========= LangChain core =========
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

# ========= RAG / Vector DB =========
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

# ========= DOCX loader =========
from docx import Document as DocxDocument


# =========================
# Streamlit 基础设置
# =========================
st.set_page_config(page_title="AI Workbench · Gemini + OpenAI RAG", layout="wide")
st.title("🧰 AI Workbench · 13 Agents + RAG（Gemini LLM + OpenAI Embedding）")


# =========================
# Env/Secrets
# =========================
load_dotenv()


def sget(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Streamlit Cloud: 优先 st.secrets
    本地：兜底 os.getenv（已 load_dotenv）
    注意：本地没有 secrets.toml 时，st.secrets 的 contains 会抛 FileNotFoundError
    """
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except FileNotFoundError:
        pass
    return os.getenv(key, default)


GOOGLE_API_KEY = sget("GOOGLE_API_KEY")
OPENAI_API_KEY = sget("OPENAI_API_KEY")
APP_PASSWORD = sget("APP_PASSWORD", "")

GEMINI_MODEL_DEFAULT = sget("GEMINI_MODEL", "gemini-1.5-pro")  # 你也可以改成 gemini-3-pro-*（以你账号可用为准）
OPENAI_EMBED_MODEL_DEFAULT = sget("OPENAI_EMBED_MODEL", "text-embedding-3-large")

if not GOOGLE_API_KEY:
    st.error("缺少 GOOGLE_API_KEY：请在 Streamlit Secrets 或本地 .env 中配置。")
    st.stop()

if not OPENAI_API_KEY:
    st.error("缺少 OPENAI_API_KEY：请在 Streamlit Secrets 或本地 .env 中配置（用于 Embedding / RAG）。")
    st.stop()


# =========================
# 简单密码门（可选）
# =========================
if APP_PASSWORD:
    if "authed" not in st.session_state:
        st.session_state.authed = False

    if not st.session_state.authed:
        st.subheader("🔒 请输入访问密码")
        pwd = st.text_input("Password", type="password", key="pwd_input")
        if st.button("进入", key="pwd_btn"):
            if pwd == APP_PASSWORD:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("密码不正确")
        st.stop()


# =========================
# 路径与目录
# =========================
PROMPTS_DIR = Path("agents") / "prompts"
KB_DIR = Path("kb")
DB_DIR = Path(".chroma_db")  # Streamlit Cloud: 容器内目录（重启可能丢，但运行中可用）

PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

AGENTS = {
    "01 人设定位 Agent": "agent_01.txt",
    "02 原创灵感讨论 Agent": "agent_02.txt",
    "03 爆款内容策划 Agent": "agent_03.txt",
    "04 文案赌注强化 Agent": "agent_04.txt",
    "05 文案开头强化 Agent": "agent_05.txt",
    "06 内容人格魅力强化 Agent": "agent_06.txt",
    "07 文案用户需求强化 Agent": "agent_07.txt",
    "08 线索转化类内容策划 Agent": "agent_08.txt",
    "09 内容影像设计 Agent": "agent_09.txt",
    "10 大纲结构组织 Agent": "agent_10.txt",
    "11 内容骨架搭建 Agent": "agent_11.txt",
    "12 整体文案改写 Agent": "agent_12.txt",
    "13 个人IP账号运营问题诊断 Agent": "agent_13.txt",
}


# =========================
# 工具函数
# =========================
def load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        return "You are a helpful assistant."
    txt = path.read_text(encoding="utf-8").strip()
    return txt or "You are a helpful assistant."


def load_docx_text(path: Path) -> str:
    doc = DocxDocument(str(path))
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts)


def load_kb_documents(agent_id: str) -> List[Document]:
    """
    从 kb/agent_XX/ 读取 docx/txt
    """
    folder = KB_DIR / agent_id
    folder.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []
    for p in folder.rglob("*"):
        if p.is_dir():
            continue
        suf = p.suffix.lower()

        if suf == ".txt":
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
        elif suf == ".docx":
            text = load_docx_text(p)
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": str(p)}))

    return docs


def extract_text(resp) -> str:
    """
    解决“乱码”：有时 resp.content 是 list[dict]（例如 [{'type':'text','text':'...'}]）
    """
    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict):
                out.append(str(block.get("text", "")))
            else:
                out.append(str(block))
        return "".join(out)
    return str(content)


def build_llm(model_name: str, temperature: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
    )


def build_embeddings(model_name: str) -> OpenAIEmbeddings:
    # OpenAI Embeddings for RAG
    return OpenAIEmbeddings(
        model=model_name,
        api_key=OPENAI_API_KEY,
    )


@st.cache_resource(show_spinner=False)
def get_vectorstore(agent_id: str, embed_model: str):
    """
    每个 agent 一个 Chroma collection
    collection_name 带 embed_model，避免你切 embedding 模型时旧库混乱
    """
    embeddings = build_embeddings(embed_model)

    persist_dir = DB_DIR / agent_id
    persist_dir.mkdir(parents=True, exist_ok=True)

    safe_model = embed_model.replace("/", "_").replace(":", "_")
    collection_name = f"kb_{agent_id}__{safe_model}"

    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    # 如果空库：写入 kb
    try:
        existing = vs._collection.count()
    except Exception:
        existing = 0

    if existing == 0:
        raw_docs = load_kb_documents(agent_id)
        if raw_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
            chunks = splitter.split_documents(raw_docs)
            vs.add_documents(chunks)
            vs.persist()

    return vs


def retrieve_context(agent_id: str, query: str, k: int, embed_model: str) -> str:
    vs = get_vectorstore(agent_id, embed_model=embed_model)
    docs = vs.similarity_search(query, k=k)
    if not docs:
        return ""

    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "kb")
        blocks.append(f"[片段{i} | 来源: {src}]\n{d.page_content}")
    return "\n\n".join(blocks)


# =========================
# Sidebar UI
# =========================
with st.sidebar:
    st.header("设置")

    st.caption("Keys 检查")
    st.write("GOOGLE_API_KEY exists:", True)
    st.write("OPENAI_API_KEY exists:", True)

    st.divider()

    agent_name = st.selectbox("选择 Agent", list(AGENTS.keys()), key="agent_select")

    # Gemini 模型（以你账号可用为准；如果 3.0 pro 的名字与你账号不同，直接改这里候选项）
    gemini_candidates = [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    ]
    gemini_default = GEMINI_MODEL_DEFAULT if GEMINI_MODEL_DEFAULT in gemini_candidates else gemini_candidates[0]
    model_name = st.selectbox("LLM（Gemini）", gemini_candidates, index=gemini_candidates.index(gemini_default), key="llm_select")

    # OpenAI Embedding 模型
    embed_candidates = [
        "text-embedding-3-large",
        "text-embedding-3-small",
    ]
    embed_default = OPENAI_EMBED_MODEL_DEFAULT if OPENAI_EMBED_MODEL_DEFAULT in embed_candidates else embed_candidates[0]
    embed_model = st.selectbox("Embedding（OpenAI）", embed_candidates, index=embed_candidates.index(embed_default), key="embed_select")

    temperature = st.slider("temperature", 0.0, 1.0, 0.3, 0.05, key="temp_slider")

    use_rag = st.toggle("启用 RAG（从 kb 检索）", value=True, key="rag_toggle")
    topk = st.slider("检索 TopK", 1, 8, 4, 1, key="topk_slider")

    if st.button("清空当前 Agent 对话", key="clear_chat_btn"):
        st.session_state.pop(f"chat::{agent_name}", None)
        st.rerun()

    st.divider()
    st.caption("📌 知识库目录：kb/agent_01 ~ kb/agent_13（docx/txt）")
    st.caption("📌 Prompt 目录：agents/prompts/agent_01.txt ~ agent_13.txt")


# =========================
# 主流程
# =========================
agent_file = AGENTS[agent_name]
system_prompt = load_prompt(agent_file)
agent_id = agent_file.replace(".txt", "")  # agent_01 ... agent_13

llm = build_llm(model_name=model_name, temperature=temperature)

chat_key = f"chat::{agent_name}"
if chat_key not in st.session_state:
    st.session_state[chat_key] = []
chat = st.session_state[chat_key]

with st.expander("查看当前 Agent 的 System Prompt（只读）", expanded=False):
    st.code(system_prompt)

# 展示历史对话
for msg in chat:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 用户输入
user_text = st.chat_input(f"正在使用：{agent_name}（可粘贴长文本）")

if user_text:
    # 1) 记录用户消息
    chat.append(HumanMessage(content=user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    # 2) RAG（强制稳定：如果 RAG 打不开就直接报错停下，避免“看似可用其实没检索”）
    rag_context = ""
    if use_rag:
        try:
            rag_context = retrieve_context(agent_id, user_text, k=topk, embed_model=embed_model)
        except Exception as e:
            st.error(f"RAG / Embedding 初始化失败：{e}")
            st.stop()

    # 3) system prompt 拼装
    sys = system_prompt
    if rag_context:
        sys = (
            system_prompt
            + "\n\n【可引用知识库片段】\n"
            + rag_context
            + "\n\n要求：如果引用了片段，请在回答中标注来源片段编号。"
        )

    messages = [SystemMessage(content=sys)] + chat

    # 4) 调用 LLM + 显示回复
    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            resp = llm.invoke(messages)
            answer = extract_text(resp)
            st.markdown(answer)

    # 5) 记录 assistant 消息
    chat.append(AIMessage(content=answer))


with st.expander("🧪 调试面板", expanded=False):
    st.write("Agent：", agent_name)
    st.write("agent_id：", agent_id)
    st.write("LLM：", model_name)
    st.write("Embedding：", embed_model)
    st.write("RAG：", use_rag, "TopK=", topk)
    st.write("KB path：", str(KB_DIR / agent_id))