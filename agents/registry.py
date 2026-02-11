from dataclasses import dataclass
from typing import Dict
from pathlib import Path


@dataclass(frozen=True)
class AgentSpec:
    key: str
    name: str
    system_prompt: str
    init_message: str


def _read_prompt(filename: str) -> str:
    p = Path(__file__).parent / "prompts" / filename
    return p.read_text(encoding="utf-8")


def get_agents() -> Dict[str, AgentSpec]:
    agents: Dict[str, AgentSpec] = {}

    agents["agent_a"] = AgentSpec(
        key="agent_a",
        name="A｜账号定位师 · 战略罗盘",
        system_prompt=_read_prompt("agent_a.txt"),
        init_message=(
            "你好，我是你的定位咨询顾问。我们先从第一步开始：\n"
            "你目前做的是什么领域？你手里有什么牌（技能/经验/资源/人设特征）？"
        ),
    )

    agents["agent_b"] = AgentSpec(
        key="agent_b",
        name="B｜选题架构师 · 爆款引擎",
        system_prompt=_read_prompt("agent_b.txt"),
        init_message=(
            "我在。把你任何模糊的灵感/碎片想法丢给我——\n"
            "一条同行爆款、一次情绪波动、一个关键词都行。\n"
            "我们从「灵感捕获」开始。"
        ),
    )

    agents["agent_c"] = AgentSpec(
        key="agent_c",
        name="C｜文案炼金师 · 结构外科",
        system_prompt=_read_prompt("agent_c.txt"),
        init_message=(
            "把你的文案初稿直接贴过来。然后只回答我一个问题：\n"
            "这条你最想拉哪一个核心数据？（点赞/评论/收藏/转发/关注，只选1个）\n"
            "我先做结构体检，再给你「手术包+整合优化版v1」。"
        ),
    )

    agents["agent_d"] = AgentSpec(
        key="agent_d",
        name="D｜影像导演 · 情绪建筑师",
        system_prompt=_read_prompt("agent_d.txt"),
        init_message=(
            "把你已经定稿的文案贴过来。\n"
            "我会先帮你设计这条视频的**情感团块与情绪暗线**——\n"
            "前半段让观众感到什么、后半段感到什么、翻转在哪里发生。\n"
            "确认暗线之后，我们再定风格，最后拆分镜。一步一步来。"
        ),
    )

    return agents
