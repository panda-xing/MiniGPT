# MiniGPT 类详解

## 一、类概览

`MiniGPT` 是一个 **Decoder-only Transformer** 语言模型，即 GPT 系列的核心架构。它的职责是：给定一段 token 序列，预测每个位置下一个 token 的概率分布。

```
输入: (B, T) 的 token ID 序列
输出: (B, T, vocab_size) 的 logits 矩阵
```

默认配置约 25M 参数，可在 8G 显存上训练。

---

## 二、构造函数 `__init__` 逐行解析

```python
class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 21128,   # 词表大小（bert-base-chinese 的 21128 个字）
        seq_len: int = 256,         # 最大序列长度
        d_model: int = 384,         # 隐藏层维度（模型宽度）
        n_heads: int = 6,           # 注意力头数
        n_layers: int = 6,          # Transformer 块的层数
        d_ff: int = 1536,           # FFN 中间层维度（通常 = 4 × d_model）
        dropout: float = 0.1,       # Dropout 概率
    ):
```

### 步骤 1：词嵌入层（Token Embedding）

```python
self.token_emb = nn.Embedding(vocab_size, d_model)
```

- **输入**: `(B, T)` 的整数 token ID
- **输出**: `(B, T, 384)` 的浮点向量
- **作用**: 将离散的 token ID 映射为连续的稠密向量，是模型理解"字"的唯一入口
- **参数量**: 21128 × 384 ≈ **8.1M**（占模型总参数的 43%）

> 类比：就像查字典，每个字对应一个 384 维的"坐标"。

### 步骤 2：位置嵌入层（Position Embedding）

```python
self.pos_emb = nn.Embedding(seq_len, d_model)
```

- **输入**: 位置索引 `0, 1, 2, ..., T-1`
- **输出**: `(T, 384)` 的位置向量
- **作用**: 为每个位置生成唯一的位置编码，让模型区分"我爱你"和"你爱我"中同一个字在不同位置的含义
- **参数量**: 256 × 384 ≈ **0.1M**
- **特点**: 可学习（Learned），与原始 Transformer 的固定 sin/cos 编码不同，通过训练自动优化

> 为什么需要位置嵌入？因为自注意力机制本身是位置无关的——它不知道谁在谁前面。

### 步骤 3：嵌入层 Dropout

```python
self.emb_dropout = nn.Dropout(dropout)
```

- **作用**: 在嵌入求和后随机置零 10% 的元素，防止过拟合
- **训练时**: 随机丢弃；**推理时**: 不丢弃（PyTorch 自动切换）

### 步骤 4：Transformer 块堆叠

```python
self.blocks = nn.ModuleList([
    TransformerBlock(d_model, n_heads, d_ff, dropout)
    for _ in range(n_layers)
])
```

- **作用**: 堆叠 6 个相同的 TransformerBlock，逐层提取更高级的语义特征
- **每一层做两件事**:
  1. **因果自注意力**: 让每个 token 与前面的 token 交互（建模上下文依赖）
  2. **前馈网络**: 对每个 token 独立做非线性变换（增加表达能力）
- **参数量**: 6 层 × (3.5M + 7.1M) / 6 ≈ **10.6M**

> 层数越深，模型能捕捉越复杂的语言现象（第1层学词法，第3层学句法，第6层学语义）。

### 步骤 5：最终层归一化

```python
self.ln_final = nn.LayerNorm(d_model)
```

- **作用**: 在最后一个 TransformerBlock 输出后做层归一化，稳定数值分布
- **为什么需要**: Pre-LN 结构中每个子层内部有 LN，但最终输出再归一化一次能提升训练稳定性

### 步骤 6：LM Head（语言模型头）

```python
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
self.lm_head.weight = self.token_emb.weight  # 权重共享（Weight Tying）
```

- **输入**: `(B, T, 384)` 的隐状态
- **输出**: `(B, T, 21128)` 的 logits（每个位置的词表得分）
- **作用**: 将高维隐状态映射回词表大小的空间，得到"下一个字是谁"的原始分数
- **权重共享**: `lm_head` 的权重矩阵和 `token_emb` 的权重矩阵指向同一块内存
  - **好处 1**: 节省 8.1M 参数
  - **好处 2**: 经验上提升生成质量——输入和输出的语义空间保持一致
- **参数量**: 0（共享后无额外参数）

> 直觉理解：如果一个词的嵌入向量在输入时"长得像"某个方向，那么在输出时它也应该在那个方向上得分高。

### 步骤 7：参数初始化

```python
self.apply(self._init_weights)
```

- **作用**: 递归遍历所有子模块，用正态分布 `N(0, 0.02)` 初始化权重，偏置初始化为 0
- **为什么**: 随机初始化的权重如果太大，训练初期梯度会爆炸；太小则梯度消失。0.02 的标准差是 GPT 系列的常用值

---

## 三、前向传播 `forward` 逐步解析

```python
def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
```

### 输入

| 参数 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `input_ids` | `(B, T)` | `long` | 一个 batch 的 token ID 序列，B=批大小，T=序列长度 |

### 步骤 1：维度提取与校验

```python
B, T = input_ids.shape
assert T <= self.seq_len, f"输入长度 {T} 超过最大序列长度 {self.seq_len}"
```

确保输入不超过模型支持的最大长度 256。

### 步骤 2：嵌入融合

```python
positions = torch.arange(T, device=input_ids.device)  # (T,) → [0, 1, 2, ..., T-1]
x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(positions))
```

执行流程：

```
input_ids: (B, T)  ──token_emb──→  (B, T, 384)  ──┐
positions: (T,)    ──pos_emb───→  (T, 384) ──广播──→ (B, T, 384) ──┤
                                                                    ├─ 逐元素相加
                                                                    ↓
                                                          (B, T, 384) ──dropout──→ x
```

- **词嵌入 + 位置嵌入 = 带位置信息的词表示**
- `pos_emb(positions)` 从 `(T, 384)` 自动广播到 `(B, T, 384)` 与 `token_emb` 相加

### 步骤 3：逐层通过 TransformerBlock

```python
for block in self.blocks:
    x = block(x)
```

每个 `TransformerBlock` 内部执行：

```
x ──┬── LN1 ── CausalSelfAttention ──┐
    │                                 ├─ 残差相加 → x
    └─────────────────────────────────┘

x ──┬── LN2 ── FeedForward ──────────┐
    │                                 ├─ 残差相加 → x
    └─────────────────────────────────┘
```

6 层堆叠后，`x` 从 `(B, T, 384)` 的原始嵌入逐渐变成蕴含深层语义的 `(B, T, 384)` 表示。

### 步骤 4：最终归一化 + 映射到词表

```python
x = self.ln_final(x)          # (B, T, 384) 归一化
return self.lm_head(x)         # (B, T, 384) → (B, T, 21128)
```

### 输出

| 返回值 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `logits` | `(B, T, vocab_size)` | `float` | 每个位置对词表中所有字的原始得分 |

> logits 不是概率！需要经过 `softmax(logits, dim=-1)` 才能得到概率分布。

---

## 四、完整数据流图

```
input_ids (B, T)
      │
      ├─ Token Embedding (21128 → 384)
      ├─ Position Embedding (256 → 384)
      │
      ▼ 相加 + Dropout
  x (B, T, 384)
      │
      ▼ ×6 层
  ┌──────────────────────────────┐
  │     TransformerBlock         │
  │                              │
  │  x ─┬─ LN ── Attention ─┐   │
  │      └──── 残差 ─────────┘   │
  │                              │
  │  x ─┬─ LN ── FFN ───────┐   │
  │      └──── 残差 ─────────┘   │
  └──────────────────────────────┘
      │
      ▼
  LN (最终归一化)
      │
      ▼
  LM Head (384 → 21128, 权重共享)
      │
      ▼
  logits (B, T, 21128)
```

---

## 五、训练时如何使用输出

```python
# train.py 中的用法
logits = model(input_ids)             # (B, T, 21128)
loss = F.cross_entropy(
    logits[:, :-1, :],                # 输入部分：第 0 ~ T-2 个位置的预测
    input_ids[:, 1:],                 # 目标部分：第 1 ~ T-1 个位置的真实 token
)
```

核心思想：**预测下一个词**。
- 位置 0 的输出预测位置 1 的字
- 位置 1 的输出预测位置 2 的字
- ...
- 位置 T-2 的输出预测位置 T-1 的字

---

## 六、推理时如何使用输出

```python
# generate.py 中的用法
logits = model(input_ids)             # (1, T, 21128)
next_logits = logits[:, -1, :]        # 只取最后一个位置的输出 (1, 21128)
probs = F.softmax(next_logits, dim=-1)# 转为概率分布

# 贪心解码：选概率最高的
next_token = probs.argmax(dim=-1)     # (1,)

# 或 Top-K 采样：从概率最高的 K 个中随机采样
# 或 Top-P 采样：从累积概率 ≤ P 的候选中采样
```

自回归生成：每次将新 token 拼接到输入末尾，再喂给模型，循环往复。

---

## 七、参数量分解

| 组件 | 计算 | 参数量 | 占比 |
|---|---|---|---|
| Token Embedding | 21128 × 384 | 8.11M | 43.3% |
| Position Embedding | 256 × 384 | 0.10M | 0.5% |
| 6 × Attention (QKV+Out) | 6 × 4 × 384² | 3.54M | 18.9% |
| 6 × FFN | 6 × 2 × 384 × 1536 | 7.08M | 37.8% |
| 6 × LayerNorm × 2 | 6 × 2 × 384 | 0.005M | <0.1% |
| Final LayerNorm | 384 × 2 | 0.0008M | <0.1% |
| LM Head | 共享 Embedding | 0 | 0% |
| **总计** | | **≈18.8M** | **100%** |

> 注：代码注释中写的 ~25M 是一个近似值，实际按上表精确计算约为 18.8M。

---

## 八、关键设计决策总结

| 设计 | 选择 | 原因 |
|---|---|---|
| 位置编码 | 可学习（Learned） | GPT-2/3 风格，实现简单，短序列效果与 sin/cos 相当 |
| 注意力掩码 | 因果掩码（Causal） | 自回归生成必须：只能看过去，不能看未来 |
| LN 位置 | Pre-LN（LN 在子层之前） | GPT-2 风格，比 Post-LN 训练更稳定 |
| 激活函数 | GELU | BERT/GPT 标配，比 ReLU 更平滑 |
| 权重共享 | LM Head ↔ Token Emb | 减少参数，提升生成质量 |
| 初始化 | N(0, 0.02) | GPT 系列标准做法，训练初期梯度稳定 |
