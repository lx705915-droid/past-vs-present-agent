# 过去的我 vs 现在的我：个人知识库辩论 Agent

这是一个可直接运行的 Python MVP：把你的历史笔记、文章、推文导出成 `.md` / `.txt`，放入 `knowledge/`，然后它会索引历史内容，并在你写新内容前生成“过去的我 vs 现在的我”自我对齐报告。

## 1. 安装

```bash
pip install -r requirements.txt
```

## 2. 配置

```bash
cp .env.example .env
```

然后打开 `.env`，填入：

```env
OPENAI_API_KEY=你的_api_key
```

## 3. 准备知识库

新建 `knowledge/` 目录，把历史文档放进去：

```bash
mkdir -p knowledge
```

示例结构：

```text
knowledge/
  2023-01-01-一篇旧文章.md
  2024-06-12-项目复盘.md
  notes.txt
```

## 4. 建立索引

```bash
python past_vs_present_agent.py index --docs ./knowledge --db ./kb_index.jsonl
```

## 5. 生成辩论报告

只输入选题：

```bash
python past_vs_present_agent.py debate \
  --topic "我现在如何看待独立开发者做 AI 产品" \
  --db ./kb_index.jsonl \
  --out ./reports/self_alignment_report.md
```

传入草稿文件：

```bash
python past_vs_present_agent.py debate \
  --topic "为什么独立开发者需要人格化产品" \
  --draft-file ./current_draft.md \
  --db ./kb_index.jsonl \
  --top-k 12 \
  --out ./reports/personality_product_report.md
```

## 6. 只检索历史观点

```bash
python past_vs_present_agent.py retrieve \
  --query "独立开发者 AI 产品 人格化" \
  --db ./kb_index.jsonl \
  --top-k 8
```

## 7. 作为 Python 模块调用

```python
from past_vs_present_agent import PastVsPresentAgent

agent = PastVsPresentAgent(db_path="./kb_index.jsonl")
agent.build_index("./knowledge")
agent.debate(
    topic="独立开发者如何做 AI 产品",
    out_path="./reports/report.md",
)
```

## 输出文件

运行后会生成：

```text
reports/self_alignment_report.md
reports/self_alignment_report.run_log.json
```

`self_alignment_report.md` 可以截图作为项目证明；`run_log.json` 可以作为多 Agent 调用与 token 消耗记录。
