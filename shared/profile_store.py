import os
from supabase import create_client, Client

# 从环境变量读取 Supabase 配置
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("❌ 未检测到 Supabase 环境变量")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def load_profile(user_id: str, project_id: str) -> dict:
    """
    读取某个 用户 + 项目 的弱共享画像
    """
    resp = (
        supabase.table("profiles")
        .select("*")
        .eq("user_id", user_id)
        .eq("project_id", project_id)
        .single()
        .execute()
    )

    if resp.data is None:
        return {}

    return resp.data.get("profile", {})


def save_profile(user_id: str, project_id: str, profile: dict):
    """
    保存（或更新）某个 用户 + 项目 的弱共享画像
    """
    payload = {
        "user_id": user_id,
        "project_id": project_id,
        "p
