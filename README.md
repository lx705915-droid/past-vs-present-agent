# 过去的我 vs 现在的我：个人知识库辩论 Agent

一个基于个人知识库的多 Agent 写作前自我对齐系统。它会把用户的历史笔记、文章和推文索引成个人观点库，并在用户写新内容前自动召回相关历史表达，让“过去的我”和“现在的我”围绕同一议题进行辩论，最后输出一份自我对齐报告。

**默认接入小米 MiMo 开放平台**（MiMo-V2.5-Pro / MiMo-V2.5），充分利用其 100 万 token 长上下文与强 Agent 能力。

## 项目解决的核心痛点

独立创作者和长期写作者最容易遇到的问题，不是没有内容，而是很难看清自己是否真的在进步：

1. **无意识重复**：写作时以为自己提出了新观点，其实只是把几个月前、甚至几年前的表达换了一种说法。
2. **观点漂移不可见**：长期写作后，观点会自然变化，但创作者很难判断这种变化是成长、妥协、退化，还是单纯换了叙事包装。
3. **个人知识库被动沉睡**：大量笔记、文章、复盘只是被存档，无法主动参与新的创作判断。
4. **缺少自我反驳机制**：普通写作助手通常只会帮用户润色和扩写，但不会用用户的历史观点来质疑当前表达。

因此，这个项目不是普通的写作工具，而是一个人格化的“认知镜子”：让过去积累的内容真正参与到现在的创作决策中。


## 核心逻辑流

系统包含长链推理和多 Agent 协作，主要分为五个阶段：

### 1. 索引 Agent

读取 `knowledge/` 目录下的 Markdown / TXT 文档，将历史笔记、文章、推文、项目复盘等内容按段落切片，并提取时间信息、文件路径和 chunk 信息。

### 2. 向量化与个人观点库

使用 Embedding 模型将历史内容向量化，生成本地 `kb_index.jsonl` 索引文件。每条记录包含：

- 原始文本片段
- 来源文件
- 时间信息
- chunk 编号
- embedding 向量

### 3. 召回 Agent

当用户输入一个新选题或草稿时，系统会把当前议题和草稿一起作为查询，在个人观点库中检索最相关的历史表达，找出“过去的我在相似主题下说过什么”。

### 4. 多 Agent 辩论（核心 · 由 MiMo-V2.5-Pro 驱动）

召回历史材料后，系统会依次启动多个 Agent：

- **过去的我 Agent**:只能基于历史材料发言，总结过去观点，并对当前议题提出补充或反驳。
- **现在的我 Agent**：基于当前选题或草稿，提炼现在的立场、新假设和潜在问题。
- **中立裁判 Agent**：比较过去观点和现在观点，判断这是观点进化、观点退化、无意识重复，还是只是换了一种说法。
- **反思 Agent**：继续追问“是什么经历导致观点变化”“这个变化是成长还是妥协”“当前表达是否只是为了维持人设”。

### 5. 自我对齐报告

最终系统会输出一份 Markdown 报告，包括：

- 召回到的历史材料
- 过去的我 Agent 分析
- 现在的我 Agent 分析
- 中立裁判 Agent 判断
- 反思 Agent 追问
- 多 Agent 运行日志与 token 使用记录

这份报告可以用于写作前自查，也可以作为项目演示材料。

## 技术栈

- Python 3.9+
- 小米 MiMo 开放平台（OpenAI 兼容接口）
- Embedding（默认 OpenAI 兼容 embedding，可替换为任意兼容服务）
- NumPy
- 本地 JSONL 向量索引
- 多 Agent 顺序协作链路

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 申请 MiMo API Key

前往 [Xiaomi MiMo 开放平台](https://platform.xiaomimimo.com) 注册并获取 API Key。

> 当前小米正在举办 MiMo Orbit 百万亿 Token 创造者激励计划，可申请免费 Token Plan：[100t.xiaomimimo.com](https://100t.xiaomimimo.com)

### 3. 配置环境变量

复制示例环境变量文件：

```bash
cp .env.example .env
```

然后在 `.env` 中填入你的 MiMo API Key：

```env
# Chat 推理：小米 MiMo（默认）
OPENAI_API_KEY=你的_MiMo_API_Key
OPENAI_BASE_URL=https://api.xiaomimimo.com/v1
OPENAI_MODEL=mimo-v2.5-pro

# Embedding：如使用 MiMo Token Plan，可单独指向其他 OpenAI 兼容 embedding 服务
EMBED_MODEL=text-embedding-3-small
EMBED_DIMS=1024
```

> 注意：不要把 `.env` 上传到 GitHub。仓库里只保留 `.env.example`。

> Base URL 说明：使用 MiMo Token Plan 订阅时为 `https://token-plan-cn.xiaomimimo.com/v1`，使用按量计费余额时为 `https://api.xiaomimimo.com/v1`。

### 4. 准备个人知识库

新建 `knowledge/` 目录，把历史笔记、文章、推文导出为 `.md` 或 `.txt` 后放进去：

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

> 文件名带日期可被自动识别为该文档的时间信息，用于"过去 vs 现在"的时间维度判断。

### 5. 建立索引

```bash
python past_vs_present_agent.py index --docs ./knowledge --db ./kb_index.jsonl
```

### 6. 只检索历史观点

```bash
python past_vs_present_agent.py retrieve \
  --query "独立开发者 AI 产品 人格化" \
  --db ./kb_index.jsonl \
  --top-k 8
```

### 7. 生成自我辩论报告

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
  --draft-file ./example_current_draft.md \
  --db ./kb_index.jsonl \
  --top-k 12 \
  --out ./reports/personality_product_report.md
```

## 作为 Python 模块调用

```python
from past_vs_present_agent import PastVsPresentAgent

agent = PastVsPresentAgent(db_path="./kb_index.jsonl")
agent.build_index("./knowledge")
agent.debate(
    topic="独立开发者如何做 AI 产品",
    draft_file="./example_current_draft.md",
    out_path="./reports/report.md",
)
```

## 输出文件

运行后会生成：

```text
reports/self_alignment_report.md
reports/self_alignment_report.run_log.json
```

其中：

- `self_alignment_report.md` 是完整的“过去的我 vs 现在的我”自我对齐报告。
- `self_alignment_report.run_log.json` 是多 Agent 调用日志，可用于查看模型、token 使用和运行过程。

## 当前进展与路线图

- [x] 核心代码完成（索引 / 召回 / 4 Agent 辩论链 / CLI / 模块调用）
- [x] 默认适配小米 MiMo 开放平台
- [x] 小规模知识库联调测试通过
- [ ] 接入 MiMo Token Plan，完成 1000+ 篇个人文档全量索引
- [ ] 30 天日常使用：写作前辩论 + 输出"观点演变"系列文章
- [ ] 扩展全模态索引（图片笔记 / 语音备忘录），适配 MiMo-V2.5 全模态能力
- [ ] 整理 MiMo 长上下文 + 多 Agent 实战案例文档，回馈 MiMo 社区

## 项目定位

这个 Agent 的核心价值不是“帮助用户写得更多”，而是“帮助用户更诚实地面对自己写过什么、现在为什么这么写，以及观点到底有没有真实变化”。

它尤其适合独立创作者、个人品牌经营者、长期写作者和独立开发者，用来发现自我重复、观点演变和长期创作中的认知惯性。

## License

MIT License
