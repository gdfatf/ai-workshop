from __future__ import annotations

from pathlib import Path
import os
import streamlit as st

from dotenv import load_dotenv

# LangChain core
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# RAG / Vector DB
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

# DOCX loader
from docx import Document as DocxDocument


# =========================
# Streamlit 基础设置
# =========================
st.set_page_config(page_title="AI Workbench · Gemini", layout="wide")
st.title("🧰 AI Workbench · 13 Agents + RAG（Gemini 3 Pro）")


# =========================
# Secrets / Env 读取
# =========================
load_dotenv()

def sget(key: str, default: str | None = None) -> str | None:
    """
    先读 Streamlit Cloud secrets；
    本地没有 secrets.toml 时，st.secrets 会抛 FileNotFoundError -> 兜底读环境变量
    """
    try:
        # st.secrets 存在但本地没文件时，会在访问/包含判断时触发 FileNotFoundError
        if key in st.secrets:
            return str(st.secrets[key])
    except FileNotFoundError:
        pass
    except Exception:
        # 极端情况下 secrets 解析异常，也兜底 env
        pass

    return os.getenv(key, default)


GOOGLE_API_KEY = sget("GOOGLE_API_KEY")
GEMINI_MODEL = sget("GEMINI_MODEL", "gemini-3-pro-preview")
# 推荐：Gemini embedding 模型名带 models/ 前缀
GEMINI_EMBEDDING_MODEL = sget("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
APP_PASSWORD = sget("APP_PASSWORD", "")

if not GOOGLE_API_KEY:
    st.error("缺少 GOOGLE_API_KEY：请在 Streamlit Cloud 的 Secrets 里配置 GOOGLE_API_KEY。")
    st.stop()


# =========================
# 简单密码门（可选）
# =========================
if APP_PASSWORD:
    if "authed" not in st.session_state:
        st.session_state.authed = False

    if not st.session_state.authed:
        st.subheader("🔒 请输入访问密码")
        pwd = st.text_input("Password", type="password")
        if st.button("进入"):
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
DB_DIR = Path(".chroma_db")  # Streamlit Cloud: 容器内可用；重启可能丢失

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


def load_kb_documents(agent_id: str) -> list[Document]:
    """
    从 kb/agent_XX/ 读取 docx/txt -> LangChain Document
    """
    folder = KB_DIR / agent_id
    folder.mkdir(parents=True, exist_ok=True)

    docs: list[Document] = []
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


def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Gemini Embedding：优先用 secrets/env 指定的 GEMINI_EMBEDDING_MODEL；
    如果不可用，则再尝试一个常见备选。
    """
    candidates = [
        GEMINI_EMBEDDING_MODEL,          # 默认 models/gemini-embedding-001
        "models/text-embedding-004",     # 某些账号/地区可用；不可用会抛错
    ]

    last_err = None
    for m in candidates:
        try:
            emb = GoogleGenerativeAIEmbeddings(
                model=m,
                google_api_key=GOOGLE_API_KEY,
            )
            _ = emb.embed_query("ping")  # 触发一次小请求校验
            return emb
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "没有可用的 embedding 模型，请检查账号权限/地区/版本。"
        f"最后错误：{last_err}"
    )


@st.cache_resource(show_spinner=False)
def get_vectorstore(agent_id: str):
    """
    每个 agent 一个 Chroma 索引（本地持久化到 .chroma_db/agent_id）
    """
    embeddings = build_embeddings()

    persist_dir = DB_DIR / agent_id
    persist_dir.mkdir(parents=True, exist_ok=True)

    vs = Chroma(
        collection_name=f"kb_{agent_id}",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    # 为空则写入
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


def retrieve_context(agent_id: str, query: str, k: int = 4) -> str:
    vs = get_vectorstore(agent_id)
    docs = vs.similarity_search(query, k=k) if vs else []
    if not docs:
        return ""

    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "kb")
        blocks.append(f"[片段{i} | 来源: {src}]\n{d.page_content}")
    return "\n\n".join(blocks)


def build_llm(model_name: str, temperature: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
    )


# =========================
# Sidebar UI
# =========================
with st.sidebar:
    st.header("设置")

    st.write("Gemini Key exists:", True)
    st.write("Default Gemini Model:", GEMINI_MODEL)
    st.write("Embedding Model:", GEMINI_EMBEDDING_MODEL)

    agent_name = st.selectbox("选择 Agent", list(AGENTS.keys()))

    model_options = [
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]
    model_name = st.selectbox(
        "模型",
        model_options,
        index=model_options.index(GEMINI_MODEL) if GEMINI_MODEL in model_options else 0,
    )

    temperature = st.slider("temperature", 0.0, 1.0, 0.3, 0.05)

    use_rag = st.toggle("启用 RAG（从 kb 检索）", value=True)
    topk = st.slider("检索 TopK", 1, 8, 4, 1)

    if st.button("清空当前 Agent 对话"):
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

for msg in chat:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

user_text = st.chat_input(f"正在使用：{agent_name}（可粘贴长文本）")

if user_text:
    chat.append(HumanMessage(content=user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    rag_context = ""
    if use_rag:
        try:
            rag_context = retrieve_context(agent_id, user_text, k=topk)
        except Exception as e:
            # embedding/向量库挂了：不让整站崩
            st.warning(f"RAG 暂不可用，已自动跳过。原因：{e}")
            rag_context = ""

    sys = system_prompt
    if rag_context:
        sys = (
            system_prompt
            + "\n\n【可引用知识库片段】\n"
            + rag_context
            + "\n\n要求：如果引用了片段，请在回答中标注来源片段编号。"
        )

    messages = [SystemMessage(content=sys)] + chat

    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
          def extract_text(resp) -> str:
    # Gemini / LangChain 兼容处理
    if isinstance(resp.content, str):
        return resp.content
    if isinstance(resp.content, list):
        return "".join(
            block.get("text", "")
            for block in resp.content
            if isinstance(block, dict)
        )
    return str(resp.content)


resp = llm.invoke(messages)
answer = extract_text(resp)

st.markdown(answer)
chat.append(AIMessage(content=answer))



# =========================
# 调试面板
# =========================
with st.expander("🧪 调试面板", expanded=False):
    st.write("当前 Agent：", agent_name)
    st.write("agent_id：", agent_id)
    st.write("模型：", model_name)
    st.write("RAG：", use_rag, "TopK=", topk)
