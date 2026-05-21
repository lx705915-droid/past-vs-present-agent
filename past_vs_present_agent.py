#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
过去的我 vs 现在的我：个人知识库辩论 Agent

默认接入小米 MiMo 开放平台（OpenAI 兼容接口），
也可通过环境变量切换到任意 OpenAI 兼容服务。

功能：
1. 将本地 knowledge/ 目录中的 .md/.txt 文档切片、向量化、建立本地 JSONL 索引。
2. 输入一个新选题或草稿，自动召回历史观点。
3. 调用多个 Agent：过去的我、现在的我、中立裁判、反思追问。
4. 输出 Markdown 自我对齐报告和 JSON 运行日志。

安装：
    pip install -r requirements.txt

配置：
    复制 .env.example 为 .env，并填入 MiMo API Key

命令行调用：
    python past_vs_present_agent.py index --docs ./knowledge --db ./kb_index.jsonl

    python past_vs_present_agent.py debate \
      --topic "我现在如何看待独立开发者做 AI 产品" \
      --db ./kb_index.jsonl \
      --out ./reports/self_alignment_report.md

作为 Python 模块调用：
    from past_vs_present_agent import PastVsPresentAgent

    agent = PastVsPresentAgent(db_path="./kb_index.jsonl")
    agent.build_index("./knowledge")
    report_path = agent.debate(topic="独立开发者如何做 AI 产品")
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from openai import OpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


# 默认接入小米 MiMo 开放平台（OpenAI 兼容接口）
# Token Plan 用户使用 token-plan-cn.xiaomimimo.com，按量计费用户使用 api.xiaomimimo.com
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.xiaomimimo.com/v1")
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_MODEL", "mimo-v2.5-pro")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DEFAULT_EMBED_DIMS = int(os.getenv("EMBED_DIMS", "1024"))
SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".mdx"}


class AgentError(RuntimeError):
    """Agent 运行错误。"""


@dataclass
class RetrievedChunk:
    rank: int
    score: float
    date: str
    path: str
    chunk_id: int
    text: str


@dataclass
class AgentRunConfig:
    chat_model: str = DEFAULT_CHAT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dims: int = DEFAULT_EMBED_DIMS
    max_chars_per_chunk: int = 1200
    chunk_overlap: int = 160
    batch_size: int = 64
    base_url: Optional[str] = DEFAULT_BASE_URL
    embed_base_url: Optional[str] = None  # embedding 可单独指向其他兼容服务


class PastVsPresentAgent:
    def __init__(
        self,
        db_path: str = "./kb_index.jsonl",
        config: Optional[AgentRunConfig] = None,
        client: Optional[OpenAI] = None,
        embed_client: Optional[OpenAI] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.config = config or AgentRunConfig()
        self.client = client or self._make_client(self.config.base_url)
        # 如果 embed_base_url 与 chat 不同，单独建一个客户端用于 embedding
        if embed_client is not None:
            self.embed_client = embed_client
        elif self.config.embed_base_url and self.config.embed_base_url != self.config.base_url:
            embed_key = os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY")
            self.embed_client = OpenAI(api_key=embed_key, base_url=self.config.embed_base_url)
        else:
            self.embed_client = self.client

    @staticmethod
    def _make_client(base_url: Optional[str] = None) -> OpenAI:
        if not os.getenv("OPENAI_API_KEY"):
            raise AgentError(
                "缺少 OPENAI_API_KEY。请在环境变量中设置，或复制 .env.example 为 .env 后填写。"
            )
        kwargs: Dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    # -------------------------
    # Indexing
    # -------------------------

    def build_index(self, docs_dir: str = "./knowledge") -> str:
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise AgentError(f"知识库目录不存在：{docs_path}")

        files = list(self._iter_source_files(docs_path))
        if not files:
            raise AgentError(f"没有在 {docs_path} 找到 .txt / .md / .mdx 文件")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        total_chunks = 0
        pending_texts: List[str] = []
        pending_metas: List[Dict[str, Any]] = []

        with self.db_path.open("w", encoding="utf-8") as f:
            for file_path in files:
                raw = self._read_text_file(file_path)
                chunks = self._chunk_text(
                    raw,
                    max_chars=self.config.max_chars_per_chunk,
                    overlap=self.config.chunk_overlap,
                )
                if not chunks:
                    continue

                doc_date = self._infer_date_from_filename(file_path)
                rel_path = str(file_path.relative_to(docs_path))

                for idx, chunk in enumerate(chunks):
                    pending_texts.append(chunk)
                    pending_metas.append(
                        {
                            "doc_id": rel_path,
                            "path": rel_path,
                            "date": doc_date,
                            "chunk_id": idx,
                            "text": chunk,
                        }
                    )

                    if len(pending_texts) >= self.config.batch_size:
                        total_chunks += self._flush_embeddings(f, pending_texts, pending_metas)
                        pending_texts = []
                        pending_metas = []

            if pending_texts:
                total_chunks += self._flush_embeddings(f, pending_texts, pending_metas)

        print(f"索引完成：{len(files)} 个文件，{total_chunks} 个 chunks，保存到 {self.db_path}")
        return str(self.db_path)

    def _flush_embeddings(
        self,
        out_file: Any,
        texts: List[str],
        metas: List[Dict[str, Any]],
    ) -> int:
        vectors = self._embed_texts(texts)
        for meta, vec in zip(metas, vectors):
            record = {**meta, "embedding": vec}
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(vectors)

    @staticmethod
    def _iter_source_files(docs_path: Path) -> Iterable[Path]:
        for path in sorted(docs_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
                yield path

    @staticmethod
    def _read_text_file(path: Path) -> str:
        for enc in ["utf-8", "utf-8-sig", "gb18030"]:
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        raise AgentError(f"无法读取文件编码：{path}")

    @staticmethod
    def _infer_date_from_filename(path: Path) -> str:
        name = path.name

        m = re.search(r"(20\d{2})[-_/\.](\d{1,2})[-_/\.](\d{1,2})", name)
        if m:
            y, mo, d = m.groups()
            return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

        m = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
        if m:
            y, mo, d = m.groups()
            return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

        ts = path.stat().st_mtime
        return time.strftime("%Y-%m-%d", time.localtime(ts))

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @classmethod
    def _chunk_text(cls, text: str, max_chars: int = 1200, overlap: int = 160) -> List[str]:
        text = cls._clean_text(text)
        if not text:
            return []

        paragraphs = re.split(r"\n\s*\n", text)
        chunks: List[str] = []
        current = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current) + len(paragraph) + 2 <= max_chars:
                current += ("\n\n" if current else "") + paragraph
            else:
                if current:
                    chunks.append(current)
                if len(paragraph) <= max_chars:
                    current = paragraph
                else:
                    step = max(1, max_chars - overlap)
                    for i in range(0, len(paragraph), step):
                        chunks.append(paragraph[i : i + max_chars])
                    current = ""

        if current:
            chunks.append(current)

        with_overlap: List[str] = []
        for i, chunk in enumerate(chunks):
            prefix = chunks[i - 1][-overlap:] if i > 0 else ""
            suffix = chunks[i + 1][:overlap] if i < len(chunks) - 1 else ""
            merged = f"{prefix}\n{chunk}\n{suffix}".strip()
            with_overlap.append(merged)

        return with_overlap

    # -------------------------
    # Retrieval
    # -------------------------

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        records = self._load_index()
        q_vec = np.array(self._embed_texts([query])[0], dtype=np.float32)
        mat = np.array([record["embedding"] for record in records], dtype=np.float32)

        denom = np.linalg.norm(mat, axis=1) * np.linalg.norm(q_vec)
        denom = np.where(denom == 0, 1e-8, denom)
        scores = mat.dot(q_vec) / denom
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[RetrievedChunk] = []
        for rank, idx in enumerate(top_indices, start=1):
            record = records[int(idx)]
            results.append(
                RetrievedChunk(
                    rank=rank,
                    score=float(scores[int(idx)]),
                    date=record.get("date", "unknown"),
                    path=record.get("path", "unknown"),
                    chunk_id=int(record.get("chunk_id", 0)),
                    text=record.get("text", ""),
                )
            )
        return results

    def _load_index(self) -> List[Dict[str, Any]]:
        if not self.db_path.exists():
            raise AgentError(f"索引文件不存在：{self.db_path}。请先运行 index 命令。")

        records: List[Dict[str, Any]] = []
        with self.db_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if not records:
            raise AgentError("索引为空，请重新运行 index 命令。")
        return records

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        kwargs: Dict[str, Any] = {
            "model": self.config.embed_model,
            "input": texts,
        }
        if self.config.embed_model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.config.embed_dims

        try:
            resp = self.embed_client.embeddings.create(**kwargs)
        except Exception as exc:
            raise AgentError(f"Embedding 调用失败：{exc}") from exc

        return [item.embedding for item in resp.data]

    # -------------------------
    # Debate
    # -------------------------

    def debate(
        self,
        topic: str,
        draft: str = "",
        draft_file: Optional[str] = None,
        top_k: int = 10,
        out_path: str = "./reports/self_alignment_report.md",
    ) -> str:
        if draft_file:
            draft = self._read_text_file(Path(draft_file))

        query = f"{topic}\n\n{draft}".strip()
        sources = self.retrieve(query, top_k=top_k)
        report, run_logs = self._make_debate_report(topic=topic, draft=draft, sources=sources)

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")

        log_path = out.with_suffix(".run_log.json")
        log_path.write_text(json.dumps(run_logs, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"报告已生成：{out}")
        print(f"运行日志已生成：{log_path}")
        return str(out)

    def _make_debate_report(
        self,
        topic: str,
        draft: str,
        sources: List[RetrievedChunk],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        source_text = self._format_sources(sources)
        base_input = f"""
# 当前议题
{topic}

# 当前草稿
{draft or "无草稿，仅有议题。"}

# 召回到的历史材料
{source_text}
""".strip()

        run_logs: List[Dict[str, Any]] = []

        past_text, log = self._call_agent(
            name="past_self_agent",
            instructions=PAST_SELF_INSTRUCTIONS,
            input_text=base_input,
            max_output_tokens=1200,
        )
        run_logs.append(log)

        present_text, log = self._call_agent(
            name="present_self_agent",
            instructions=PRESENT_SELF_INSTRUCTIONS,
            input_text=base_input,
            max_output_tokens=1200,
        )
        run_logs.append(log)

        judge_input = f"""
# 当前议题
{topic}

# 历史材料
{source_text}

# 过去的我 Agent 输出
{past_text}

# 现在的我 Agent 输出
{present_text}
""".strip()

        judge_text, log = self._call_agent(
            name="judge_agent",
            instructions=JUDGE_INSTRUCTIONS,
            input_text=judge_input,
            max_output_tokens=1500,
        )
        run_logs.append(log)

        reflection_input = f"""
# 当前议题
{topic}

# 过去的我
{past_text}

# 现在的我
{present_text}

# 裁判分析
{judge_text}
""".strip()

        reflection_text, log = self._call_agent(
            name="reflection_agent",
            instructions=REFLECTION_INSTRUCTIONS,
            input_text=reflection_input,
            max_output_tokens=1200,
        )
        run_logs.append(log)

        report = f"""# 过去的我 vs 现在的我：自我对齐报告

## 议题

{topic}

---

## 一、召回到的历史材料

{source_text}

---

## 二、过去的我 Agent

{past_text}

---

## 三、现在的我 Agent

{present_text}

---

## 四、中立裁判 Agent

{judge_text}

---

## 五、反思 Agent 追问

{reflection_text}

---

## 六、运行记录

```json
{json.dumps(run_logs, ensure_ascii=False, indent=2)}
```
"""
        return report, run_logs

    @staticmethod
    def _format_sources(sources: List[RetrievedChunk], max_chars_each: int = 700) -> str:
        if not sources:
            return "没有召回到历史材料。"

        lines: List[str] = []
        for i, source in enumerate(sources, start=1):
            text = source.text.replace("\n", " ").strip()
            if len(text) > max_chars_each:
                text = text[:max_chars_each] + "..."
            lines.append(
                f"[S{i}] date={source.date} file={source.path} "
                f"chunk={source.chunk_id} score={source.score:.4f}\n{text}"
            )
        return "\n\n".join(lines)

    def _call_agent(
        self,
        name: str,
        instructions: str,
        input_text: str,
        max_output_tokens: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        使用 Chat Completions 接口调用 Agent。
        MiMo 开放平台 OpenAI 兼容，因此使用 chat.completions.create
        以获得最佳跨服务兼容性。
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.config.chat_model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input_text},
                ],
                max_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise AgentError(f"{name} 调用失败：{exc}") from exc

        text = self._extract_response_text(resp)
        return text, {
            "agent": name,
            "model": self.config.chat_model,
            "usage": self._usage_to_dict(resp),
        }

    @staticmethod
    def _extract_response_text(resp: Any) -> str:
        try:
            choice = resp.choices[0]
            content = choice.message.content
            if content:
                return content.strip()
        except Exception:
            pass
        return str(resp)

    @staticmethod
    def _usage_to_dict(resp: Any) -> Dict[str, Any]:
        try:
            usage = getattr(resp, "usage", None)
            if usage is None:
                return {}
            if hasattr(usage, "model_dump"):
                return usage.model_dump()
            if isinstance(usage, dict):
                return usage
            return dict(usage)
        except Exception:
            return {}


PAST_SELF_INSTRUCTIONS = """
你是"过去的我" Agent。
你只能基于召回到的历史材料发言，不能编造经历、项目、数据或观点。

你的任务：
1. 总结过去的我在这个议题上的核心立场。
2. 模拟过去的我的思考方式，提出对当前议题/草稿的回应。
3. 明确引用证据编号，例如 [S1]、[S3]。
4. 如果历史材料不足，要直接说"不足以判断"。

输出结构：
- 过去立场摘要
- 过去的我会如何反驳或补充
- 证据来源
""".strip()


PRESENT_SELF_INSTRUCTIONS = """
你是"现在的我" Agent。
你需要基于当前议题和当前草稿，提炼"现在的我"的立场。
不要讨好过去的我，不要为了显得进步而强行制造变化。

你的任务：
1. 提炼当前观点。
2. 指出当前观点里的新假设、新经验、新判断。
3. 指出当前草稿可能存在的空话、重复、证据不足。

输出结构：
- 当前立场摘要
- 新增判断
- 当前草稿的问题
""".strip()


JUDGE_INSTRUCTIONS = """
你是中立裁判 Agent。
你需要比较"过去的我"和"现在的我"的观点关系。

重点判断：
1. 这是观点进化、观点倒退、表述重复，还是只是换了说法？
2. 当前草稿有没有无意识重复过去表达？
3. 当前观点相比过去，新增了哪些真实信息？
4. 有哪些矛盾需要作者面对？

输出结构：
- 总体判断
- 重复风险
- 观点演变
- 关键矛盾
- 建议作者删掉/保留/重写的部分
""".strip()


REFLECTION_INSTRUCTIONS = """
你是反思 Agent。
你的任务不是总结，而是追问。

你要逼作者回答：
1. 是什么经历让观点发生变化？
2. 这个变化是成长、妥协、懒惰，还是叙事包装？
3. 有没有为了显得成熟而否定过去？
4. 有没有为了保持人设而拒绝承认变化？
5. 下一版文章应该如何更诚实？

输出 8 到 12 个尖锐但具体的问题。
每个问题都要能直接帮助作者改稿。
""".strip()


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="个人知识库：过去的我 vs 现在的我 辩论 Agent")
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help="用于多 Agent 推理的模型")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="用于向量化的模型")
    parser.add_argument("--embed-dims", type=int, default=DEFAULT_EMBED_DIMS, help="embedding 维度")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Chat 接口 base_url（默认 MiMo）")
    parser.add_argument("--embed-base-url", default=None, help="Embedding 接口 base_url（不填则与 chat 一致）")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="索引个人知识库")
    p_index.add_argument("--docs", default="./knowledge", help="知识库目录")
    p_index.add_argument("--db", default="./kb_index.jsonl", help="索引输出路径")
    p_index.add_argument("--batch-size", type=int, default=64, help="Embedding 批量大小")
    p_index.add_argument("--max-chars", type=int, default=1200, help="每个 chunk 最大字符数")
    p_index.add_argument("--overlap", type=int, default=160, help="chunk 重叠字符数")

    p_retrieve = sub.add_parser("retrieve", help="只检索历史材料，不生成报告")
    p_retrieve.add_argument("--query", required=True, help="检索问题")
    p_retrieve.add_argument("--db", default="./kb_index.jsonl", help="索引路径")
    p_retrieve.add_argument("--top-k", type=int, default=10)

    p_debate = sub.add_parser("debate", help="生成自我辩论报告")
    p_debate.add_argument("--topic", required=True, help="当前议题/选题")
    p_debate.add_argument("--draft", default="", help="当前草稿文本")
    p_debate.add_argument("--draft-file", default=None, help="当前草稿文件")
    p_debate.add_argument("--db", default="./kb_index.jsonl", help="索引路径")
    p_debate.add_argument("--top-k", type=int, default=10)
    p_debate.add_argument("--out", default="./reports/self_alignment_report.md")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = make_arg_parser()
    args = parser.parse_args(argv)

    config = AgentRunConfig(
        chat_model=args.chat_model,
        embed_model=args.embed_model,
        embed_dims=args.embed_dims,
        batch_size=getattr(args, "batch_size", 64),
        max_chars_per_chunk=getattr(args, "max_chars", 1200),
        chunk_overlap=getattr(args, "overlap", 160),
        base_url=args.base_url,
        embed_base_url=args.embed_base_url,
    )

    try:
        agent = PastVsPresentAgent(db_path=getattr(args, "db", "./kb_index.jsonl"), config=config)

        if args.cmd == "index":
            agent.build_index(docs_dir=args.docs)
            return 0

        if args.cmd == "retrieve":
            chunks = agent.retrieve(args.query, top_k=args.top_k)
            print(agent._format_sources(chunks, max_chars_each=1200))
            return 0

        if args.cmd == "debate":
            agent.debate(
                topic=args.topic,
                draft=args.draft,
                draft_file=args.draft_file,
                top_k=args.top_k,
                out_path=args.out,
            )
            return 0

        parser.print_help()
        return 1

    except AgentError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
