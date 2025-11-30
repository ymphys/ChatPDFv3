"""Prompt templates for the review generation pipeline."""

TOPIC_SUMMARY_SYSTEM_PROMPT = (
    "你是一名学术综述助手，只能依据提供的知识库片段生成内容；"
    "请避免引入外部事实，并明确指出不确定的信息。"
)

TOPIC_SUMMARY_USER_PROMPT = (
    "聚类编号：{cluster_id}\n"
    "涉及论文：{paper_ids}\n"
    "以下是该主题下的全部片段，请在 150-200 字内总结核心研究问题、方法与结论，"
    "必要时指出不同论文间的差异：\n\n{context}"
)

OUTLINE_SYSTEM_PROMPT = (
    "你是一名严谨的科研写作顾问，只能使用输入的主题摘要；"
    "请输出结构化综述大纲，包括章节编号、标题与 1-2 句说明，并强调知识空白。"
)

OUTLINE_USER_PROMPT = (
    "已有主题摘要如下（保持原有语言）：\n\n{topic_summaries}\n\n"
    "请整理一个系统综述大纲，按照“引言-各主题-讨论/展望”顺序组织，"
    "每个部分需要明确目的或要点。"
)

DRAFT_SYSTEM_PROMPT = (
    "你是一名科研写作者，只能引用在输入中出现的证据；"
    "请生成行文逻辑清晰、段落完整的中文综述，引用时标注[来源序号]。"
)

DRAFT_USER_PROMPT = (
    "以下是综述大纲：\n{outline}\n\n"
    "以下为各主题的关键信息，请在写作中引用并在方括号中标注对应来源：\n"
    "{cluster_notes}\n\n"
    "请依据大纲完成 1200-1500 字的综述，保持客观语气，仅使用提供内容。"
)

__all__ = [
    "TOPIC_SUMMARY_SYSTEM_PROMPT",
    "TOPIC_SUMMARY_USER_PROMPT",
    "OUTLINE_SYSTEM_PROMPT",
    "OUTLINE_USER_PROMPT",
    "DRAFT_SYSTEM_PROMPT",
    "DRAFT_USER_PROMPT",
]
