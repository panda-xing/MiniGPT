# Generate 详解 — 四种解码策略的文本生成

## 一、文件概览

`generate.py` 是预训练模型的推理入口，实现了从训练好的 MiniGPT 生成文本的功能。核心价值在于**四种解码策略的对比实现**，帮助理解确定性与多样性之间的权衡。

```
输入: 提示文本 (prompt) + 训练好的模型检查点
输出: 模型续写的文本
```

---

## 二、辅助函数

### 2.1 get_tokenizer — 加载分词器

```python
def get_tokenizer():
    local_path = Path(r"D:\BaiduNetdiskDownload\bert-base-chinese\bert-base-chinese")
    if local_path.exists():
        return BertTokenizerFast.from_pretrained(str(local_path))
    return BertTokenizerFast.from_pretrained("bert-base-chinese")
```

优先加载本地 `bert-base-chinese`，没有则从 HuggingFace 下载。生成时必须用和训练时**完全相同的分词器**，否则 token ID 对不上。

### 2.2 load_model — 加载训练好的模型

```python
def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(vocab_size=ckpt["vocab_size"], seq_len=ckpt["seq_len"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()   # 切换到评估模式，关闭 Dropout
    return model, ckpt["seq_len"]
```

| 步骤 | 说明 |
|---|---|
| `torch.load(..., map_location=device)` | 直接加载到目标设备，避免先加载到 CPU 再搬运 |
| `build_model(vocab_size, seq_len)` | 必须与训练时的配置一致 |
| `model.load_state_dict(...)` | 填入训练好的权重 |
| `model.eval()` | 关闭 Dropout，保证推理结果确定性 |

> 为什么推理时必须 `model.eval()`？训练时 Dropout 随机丢弃 10% 的输出，推理时应该用全部信息。

### 2.3 decode_text — ID 转文本

```python
def decode_text(tokenizer, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(ids.tolist(), skip_special_tokens=skip_special_tokens)
```

将 token ID 序列转回可读文本。`skip_special_tokens=True` 过滤掉 `[CLS]`、`[SEP]` 等特殊标记。

---

## 三、generate 函数 — 核心生成逻辑（逐行解析）

### 3.1 函数签名

```python
@torch.no_grad()   # 装饰器：整个函数不计算梯度，节省显存
def generate(
    model,                              # MiniGPT 模型
    input_ids: torch.Tensor,            # (1, T) 提示文本的 token ID
    max_new_tokens: int = 80,           # 最多生成多少个新 token
    strategy: str = "greedy",           # 解码策略
    temperature: float = 1.0,           # 温度参数
    top_k: int = 50,                    # Top-K 的 K 值
    top_p: float = 0.9,                # Top-P 的 P 值
) -> torch.Tensor:                      # 返回 (total_len,) 的 token ID
```

### 3.2 自回归循环骨架

```python
generated = input_ids.clone()  # (1, T) 复制一份，不修改原始输入

for _ in range(max_new_tokens):
    # 1. 截取上下文（不超过模型最大长度）
    context = generated[:, -seq_len:]

    # 2. 前向传播
    logits = model(context)            # (1, T, V)
    next_logits = logits[0, -1, :]     # (V,) 只取最后一个位置

    # 3. 按策略选 next token
    next_token = ...                    # (1,)

    # 4. 拼接到序列末尾
    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)  # (1, T+1)

return generated[0]  # (total_len,)
```

关键理解：**自回归 = 每次把新 token 拼回去，再喂给模型预测下一个**。

```
第1步: [中,国,的,首,都,是] → 预测 "北"
第2步: [中,国,的,首,都,是,北] → 预测 "京"
第3步: [中,国,的,首,都,是,北,京] → 预测 "是"
...
```

### 3.3 上下文截断

```python
context = generated[:, -seq_len:]
```

如果 `generated` 长度超过模型最大序列长度（256），只取最后 256 个 token 作为输入。这叫**滑动窗口**策略。

> 为什么不能输入超过 256 的序列？因为位置嵌入只有 256 个位置，超出就没有对应的位置编码了。

### 3.4 为什么只取最后一个位置的 logits？

```python
next_logits = logits[0, -1, :]  # (V,)
```

模型输出 `(1, T, V)`，每个位置都预测下一个 token。但我们只需要**基于完整上下文的预测**，即最后一个位置。其他位置的预测是基于不完整上下文的，对生成无用。

---

## 四、四种解码策略详解

### 4.1 Greedy Decoding（贪心解码）

```python
if strategy == "greedy":
    next_token = next_logits.argmax(dim=-1, keepdim=True)
```

**原理**: 每步选概率最高的那个 token，不做任何随机。

```
logits:     [2.1, 5.3, 1.0, 0.5, ...]
                ↓ softmax
probs:      [0.04, 0.90, 0.02, 0.01, ...]
                           ↓ argmax
选择:       token_id=1  （概率 0.90）
```

| 优点 | 缺点 |
|---|---|
| 确定性，相同输入永远相同输出 | 容易陷入重复循环（"的的的..."） |
| 实现最简单 | 缺乏多样性，输出呆板 |
| 适合事实性问答 | 不适合创意生成 |

> 示例：`"中国的首都是"` → `"北京。北京是中国的首都，北京..."`（开始重复）

### 4.2 Temperature Sampling（温度采样）

```python
next_logits = next_logits / max(temperature, 1e-8)
# strategy == "temperature" 时，只做缩放，不做 top-k/p 截断
probs = F.softmax(next_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

**原理**: 在 softmax 之前，将 logits 除以温度 T。T 改变概率分布的"尖锐程度"。

```
原始 logits:    [2.0, 5.0, 1.0]
                 ↓ / T

T=0.5 (更尖锐): [4.0, 10.0, 2.0]  → softmax → [0.01, 0.98, 0.01]  趋近贪心
T=1.0 (不变):   [2.0, 5.0, 1.0]   → softmax → [0.05, 0.84, 0.11]  原始分布
T=2.0 (更平滑): [1.0, 2.5, 0.5]   → softmax → [0.15, 0.67, 0.18]  更随机
T→0:            接近 argmax（贪心）
T→∞:            接近均匀分布（完全随机）
```

| 温度 | 效果 | 适用场景 |
|---|---|---|
| 0.1~0.5 | 非常保守，接近贪心 | 代码生成、事实问答 |
| 0.7~1.0 | 适度随机 | 通用文本生成 |
| 1.5~2.0 | 非常随机，甚至混乱 | 创意写作（需谨慎） |

> 代码中默认 temperature=0.8，略偏保守，适合中文生成。

### 4.3 Top-K Sampling（Top-K 采样）

```python
next_logits = next_logits / max(temperature, 1e-8)

# 将低于第 K 大的 logits 置为 -inf
values, _ = torch.topk(next_logits, top_k)   # 取最大的 K 个值
threshold = values[-1]                         # 第 K 大的值作为阈值
next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

probs = F.softmax(next_logits, dim=-1)         # 被置为 -inf 的位置概率为 0
next_token = torch.multinomial(probs, num_samples=1)
```

**原理**: 只保留概率最高的 K 个候选，其余全部淘汰，然后从这 K 个中按概率采样。

```
原始分布（假设 K=3）:
  token_0: 0.05
  token_1: 0.50  ┐
  token_2: 0.30  ├─ 保留（Top-3）
  token_3: 0.10  ┘
  token_4: 0.03
  token_5: 0.02

截断后重新归一化:
  token_1: 0.50 / 0.90 = 0.556
  token_2: 0.30 / 0.90 = 0.333
  token_3: 0.10 / 0.90 = 0.111
  其余: 0（被排除）
```

| 优点 | 缺点 |
|---|---|
| 避免选到极低概率的离谱 token | K 是固定的，无法适应分布变化 |
| 实现直观 | 分布集中时 K 太大（浪费），分布平坦时 K 太小（截断太多） |

> K=50 是常用值。K=1 等价于贪心，K=vocab_size 等价于纯温度采样。

### 4.4 Top-P / Nucleus Sampling（核采样）

```python
next_logits = next_logits / max(temperature, 1e-8)

# 按概率从大到小排序
sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

# 找到累积概率超过 p 的位置
sorted_indices_to_remove = cumprobs > top_p

# 右移一位：保留恰好跨过阈值的那个 token（至少保留 1 个）
sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
sorted_indices_to_remove[0] = False

# 映射回原始顺序，将排除的 token 置为 -inf
indices_to_remove = sorted_indices[sorted_indices_to_remove]
next_logits[indices_to_remove] = float("-inf")

probs = F.softmax(next_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

**原理**: 按概率从大到小累加，累积到 p 时停止，只在累积集合（nucleus）中采样。

```
原始分布（假设 p=0.9）:
  token_1: 0.50  ──── 累积 0.50 ──┐
  token_2: 0.25  ──── 累积 0.75 ──┤ 保留（nucleus）
  token_3: 0.15  ──── 累积 0.90 ──┘ ← 刚好达到 0.9
  token_4: 0.06  ──── 累积 0.96 ──┐ 排除
  token_5: 0.04  ──── 累积 1.00 ──┘
```

**右移一位的精妙之处**：

```python
sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
sorted_indices_to_remove[0] = False
```

这行代码确保：即使第一个 token 的概率就超过 p，也不会被排除（至少保留 1 个候选）。

```
移位前: [False, True, True, True, True]   ← cumprobs > 0.9 的位置
移位后: [False, False, True, True, True]  ← token_3 被保留
```

| 优点 | 缺点 |
|---|---|
| 自适应候选数量：分布集中时少选，分布平坦时多选 | 实现比 Top-K 略复杂 |
| 目前最主流的解码策略 | p 值选择仍需经验 |
| 避免了 Top-K 的固定 K 问题 | — |

> p=0.9 是最常用值。p=0 等价于贪心，p=1.0 等价于纯温度采样。

### 4.5 四种策略对比表

| 策略 | 核心操作 | 候选集大小 | 确定性 | 多样性 | 速度 |
|---|---|---|---|---|---|
| Greedy | argmax | 1 | 完全确定 | 无 | 最快 |
| Temperature | logits / T + 采样 | 全部词表 | 随机 | T 控制 | 快 |
| Top-K | 保留前 K + 采样 | 固定 K | 随机 | 中等 | 快 |
| Top-P | 累积概率 ≤ p + 采样 | 自适应 | 随机 | 中高 | 稍慢（需排序） |

---

## 五、compare_strategies 函数 — 四策略并排对比

```python
def compare_strategies(model, tokenizer, prompt, max_new_tokens, device):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    strategies = [
        ("Greedy",       dict(strategy="greedy")),
        ("Temperature",  dict(strategy="temperature", temperature=0.8)),
        ("Top-K (K=50)", dict(strategy="top_k", temperature=0.8, top_k=50)),
        ("Top-P (p=0.9)",dict(strategy="top_p", temperature=0.8, top_p=0.9)),
    ]

    for name, kwargs in strategies:
        out_ids = generate(model, input_ids, max_new_tokens=max_new_tokens, **kwargs)
        new_ids = out_ids[prompt_len:]               # 只取生成部分
        generated_text = decode_text(tokenizer, new_ids)
        print(f"【{name}】")
        print(f"{prompt}{generated_text}")
```

注意三种采样策略都使用了 `temperature=0.8`，这使对比更公平——差异仅来自截断策略。

> `out_ids[prompt_len:]`：从完整输出中切掉 prompt 部分，只展示新生成的文本。

---

## 六、main 函数 — 命令行入口

```python
parser.add_argument("--checkpoint", type=str, default=None)        # 模型路径
parser.add_argument("--prompt", type=str, default="中国的首都是")   # 提示文本
parser.add_argument("--max_new_tokens", type=int, default=80)      # 生成长度
parser.add_argument("--strategy", type=str, default="top_p",       # 解码策略
                    choices=["greedy", "temperature", "top_k", "top_p"])
parser.add_argument("--temperature", type=float, default=0.8)      # 温度
parser.add_argument("--top_k", type=int, default=50)               # K 值
parser.add_argument("--top_p", type=float, default=0.9)            # P 值
parser.add_argument("--compare", action="store_true")              # 对比模式
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--checkpoint` | outputs/checkpoints/best_model.pt | 模型检查点路径 |
| `--prompt` | "中国的首都是" | 生成起始文本 |
| `--max_new_tokens` | 80 | 最多生成 token 数 |
| `--strategy` | top_p | 解码策略（推荐 top_p） |
| `--temperature` | 0.8 | 温度（< 1 偏保守，> 1 偏随机） |
| `--top_k` | 50 | Top-K 的候选数 |
| `--top_p` | 0.9 | Top-P 的累积概率阈值 |
| `--compare` | False | 并排对比四种策略 |

使用示例：

```bash
# 单策略生成
python generate.py --prompt "人工智能的未来" --strategy top_p

# 对比四种策略
python generate.py --compare --prompt "在很久很久以前" --max_new_tokens 100
```

---

## 七、自回归生成的完整数据流

```
Prompt: "中国的首都是"
        │
        ▼ tokenizer.encode
input_ids: [1010, 862, 576, 4638, 3221, 3221, 6568]
  shape: (1, 7)
        │
        ▼ ─────────── 第1次迭代 ───────────
        │
  context = input_ids[:, -256:]     (1, 7)
        │
        ▼ model.forward()
  logits (1, 7, 21128)
        │
        ▼ 取最后一个位置
  next_logits (21128,)
        │
        ▼ Top-P 策略筛选 + 采样
  next_token = tensor([1266])       ← "北"
        │
        ▼ 拼接
  generated (1, 8): [..., 1266]
        │
        ▼ ─────────── 第2次迭代 ───────────
        │
  context = generated[:, -256:]     (1, 8)
        │
        ▼ model.forward()
  logits (1, 8, 21128)
        │
        ▼ 取最后一个位置
  next_logits (21128,)
        │
        ▼ Top-P 策略筛选 + 采样
  next_token = tensor([833])        ← "京"
        │
        ▼ 拼接
  generated (1, 9): [..., 1266, 833]
        │
       ...重复 max_new_tokens 次...
        │
        ▼ tokenizer.decode
"中国的首都是北京..."
```

---

## 八、Top-K vs Top-P：为什么 Top-P 更好

用一个具体例子说明：

### 场景 A：模型非常确定（分布尖锐）

```
token_1: 0.92
token_2: 0.03
token_3: 0.02
...
token_50: 0.0001
```

- **Top-K=50**: 保留 50 个候选，但前 2 个就占了 95% 概率，后面 48 个几乎不可能被选中 → **浪费计算**
- **Top-P=0.9**: 只需保留 token_1 就够了（累积概率已达 0.92）→ **精确高效**

### 场景 B：模型很不确定（分布平坦）

```
token_1: 0.10
token_2: 0.09
token_3: 0.08
...
token_50: 0.01
```

- **Top-K=50**: 保留 50 个，可能不够（还有 token_51~100 也有不少概率）→ **截断太多**
- **Top-P=0.9**: 需要保留更多 token 才能累积到 0.9 → **自适应扩展**

> 结论：Top-P 根据分布形状自动调整候选集大小，是当前最推荐的解码策略。

---

## 九、关键设计决策总结

| 设计 | 选择 | 原因 |
|---|---|---|
| 梯度计算 | `@torch.no_grad()` 装饰器 | 推理不需要梯度，节省约 50% 显存 |
| 上下文管理 | 滑动窗口（截取最后 seq_len 个） | 处理超长序列时不报错 |
| Temperature 位置 | 在 softmax 之前除以 T | 数学等价于改变分布的"温度"，直接操作 logits 更高效 |
| Top-P 排序 | `torch.sort(descending=True)` | 需要从大到小累积，排序一次即可 |
| 至少保留 1 个 | 右移掩码技巧 | 防止极端情况下所有 token 被排除 |
| 默认策略 | top_p | 自适应，当前工业界最常用 |
| 默认温度 | 0.8 | 略偏保守，中文生成效果更自然 |
