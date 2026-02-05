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

    agents["agent_01"] = AgentSpec(
        key="agent_01",
        name="01 人设定位咨询顾问",
        system_prompt=_read_prompt("agent_01.txt"),
        init_message="你好，我是你的定位咨询顾问。先别急着想文案，让我们先从你手里的牌看起。"
    )

    agents["agent_02"] = AgentSpec(
        key="agent_02",
        name="02 原创灵感讨论（爆款内容合伙人）",
        system_prompt=_read_prompt("agent_02.txt"),
        init_message="我在。把你任何模糊的感想/灵感丢给我，我们先从第一步「情绪挖掘」开始：它来自哪一次情绪波动或哪条同领域热门？"
    )

    agents["agent_03"] = AgentSpec(
        key="agent_03",
        name="03 爆款内容策划（14维创意选题）",
        system_prompt=_read_prompt("agent_03.txt"),
        init_message="你好，我是你的创意选题顾问。你最近想写什么话题/处于什么赛道？先给我一个很粗的想法就行。"
    )

    agents["agent_04"] = AgentSpec(
        key="agent_04",
        name="04 文案赌注强化（情绪刺点）",
        system_prompt=_read_prompt("agent_04.txt"),
        init_message="你好！我是你的情绪刺点优化师。请把你的视频脚本初稿发给我，并告诉我：这条视频最想拉动的核心数据是哪一个（点赞/评论/收藏/转发/关注）？"
    )

    agents["agent_05"] = AgentSpec(
        key="agent_05",
        name="05 文案开头强化（钩子）",
        system_prompt=_read_prompt("agent_05.txt"),
        init_message="请发送你的【选题】和【文案】，我将为你定制爆款钩子（含画面建议+台词开头+设计逻辑）。"
    )

    agents["agent_06"] = AgentSpec(
        key="agent_06",
        name="06 内容人格魅力强化（人设炼金术）",
        system_prompt=_read_prompt("agent_06.txt"),
        init_message="请发送你的原始文案，我将为你注入灵魂与人设。"
    )

    agents["agent_07"] = AgentSpec(
        key="agent_07",
        name="07 用户需求强化（流量漏斗思维）",
        system_prompt=_read_prompt("agent_07.txt"),
        init_message="把你现在的原始选题/标题发我。我先诊断它“漏斗口有多大、钩子够不够精准、是否足够新鲜”。"
    )

    agents["agent_08"] = AgentSpec(
        key="agent_08",
        name="08 线索转化内容策划（流量变现策略）",
        system_prompt=_read_prompt("agent_08.txt"),
        init_message="你好，我是你的流量变现策略顾问。先告诉我：你目前做的具体领域是什么？你是个人单打独斗还是有团队？"
    )

    agents["agent_09"] = AgentSpec(
        key="agent_09",
        name="09 内容影像设计（用画面讲故事）",
        system_prompt=_read_prompt("agent_09.txt"),
        init_message="把你的文案/脚本段落发我。我会先定风格与基调，然后给你按段落输出可拍的分镜表。"
    )

    agents["agent_10"] = AgentSpec(
        key="agent_10",
        name="10 大纲结构组织（材料组织思维）",
        system_prompt=_read_prompt("agent_10.txt"),
        init_message="你好，我是你的大纲架构教练。请告诉我你现在的主题是什么，以及你手头已经有了哪些主要内容/想法？"
    )

    agents["agent_11"] = AgentSpec(
        key="agent_11",
        name="11 内容骨架搭建（骨架精细化处理）",
        system_prompt=_read_prompt("agent_11.txt"),
        init_message="先把你的文案发我，然后告诉我：这条视频你最核心想达到的目的/想触发观众的哪种情绪是什么？"
    )

    agents["agent_12"] = AgentSpec(
        key="agent_12",
        name="12 整体文案改写（文字语言基本功）",
        system_prompt=_read_prompt("agent_12.txt"),
        init_message="你好！我是你的文案基本功教练。请发送你的文案初稿，我们开始第一步：结构诊断。"
    )

    agents["agent_13"] = AgentSpec(
        key="agent_13",
        name="13 IP运营问题诊断（数据复盘与规划）",
        system_prompt=_read_prompt("agent_13.txt"),
        init_message="你好！做IP是一场马拉松，我们先看看你跑到哪里了：你最近几条视频的平均播放量大概在什么区间（5000以下/5万左右/30万以上）？"
    )

    return agents
