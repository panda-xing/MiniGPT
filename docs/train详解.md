# Train 详解 — GPT 预训练主循环

## 一、文件概览

`train.py` 是整个项目的训练入口，实现了一个完整的自回归语言模型预训练流程。它把数据加载、模型构建、优化器配置、训练循环、验证评估、检查点保存串成一条线。

```
输入: 预处理好的 .pt 数据集（由 prepare_dataset.py 生成）
输出: 训练好的模型检查点 + 训练日志
```

---

## 二、目录结构与常量

```python
BASE_DIR = Path(__file__).parent.parent   # 项目根目录
DATA_DIR = BASE_DIR / "data"              # 数据目录
OUTPUT_DIR = BASE_DIR / "outputs"         # 输出根目录
CKPT_DIR = OUTPUT_DIR / "checkpoints"     # 检查点目录
LOG_PATH = OUTPUT_DIR / "training_log.jsonl"  # 训练日志
```

对应文件系统：

```
week5/
├── data/
│   ├── train_seq256.pt    ← 训练集
│   └── val_seq256.pt      ← 验证集
└── outputs/
    ├── checkpoints/
    │   ├── epoch1_pplXXXX.X.pt
    │   ├── epoch2_pplXXXX.X.pt
    │   └── best_model.pt   ← 最优模型
    └── training_log.jsonl  ← 每轮记录
```

---

## 三、TokenDataset 类 — 数据加载

### 核心思想

`prepare_dataset.py` 已经将所有 token 拼接成长序列，按 `seq_len+1` 切成固定长度的块，每块形状为 `(257,)`。

```python
class TokenDataset(Dataset):
    def __init__(self, pt_path: Path):
        ckpt = torch.load(pt_path, weights_only=True)
        self.data = ckpt["data"]           # (N, 257)
        self.vocab_size = ckpt["vocab_size"]
        self.seq_len = ckpt["seq_len"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]             # (257,)
        return chunk[:-1], chunk[1:]       # input(256), target(256)
```

### 为什么要 +1？

自回归训练的核心：**用前面的 token 预测下一个 token**。

```
原始块:  [t0, t1, t2, t3, ..., t255, t256]   ← 长度 257
              │                    │       │
input:    [t0, t1, t2, ..., t255]             ← 前 256 个
target:   [t1, t2, t3, ..., t256]             ← 后移一位

位置0看t0预测t1，位置1看t0,t1预测t2，...，位置255看全部预测t256
```

> 一个 257 长度的块同时提供 256 个训练信号，数据利用率极高。

### DataLoader 配置

```python
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,    # 默认 32
    shuffle=True,             # 训练集打乱，防止顺序偏差
    num_workers=num_workers,  # 数据加载线程数
    pin_memory=True,          # GPU 训练时锁页内存，加速 CPU→GPU 传输
)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
```

| 参数 | 训练集 | 验证集 | 原因 |
|---|---|---|---|
| `shuffle` | True | False | 训练需随机化；验证需可复现 |
| `pin_memory` | True（GPU时） | True（GPU时） | 避免数据从 CPU 到 GPU 时的额外拷贝 |

---

## 四、compute_ppl 函数 — 困惑度计算

### 什么是 PPL？

PPL（Perplexity，困惑度）= 模型对验证集的"惊讶程度"。

```
PPL = exp(平均交叉熵损失)
```

| PPL 值 | 含义 |
|---|---|
| ≈ 21128 | 随机猜测（等于词表大小） |
| 1000+ | 模型刚开始学习 |
| 100-500 | 有一定语言能力 |
| < 100 | 较好的语言模型 |

> 直觉：PPL=100 意味着模型平均在每个位置"犹豫"100 个候选词；PPL=20 则只需在 20 个中选。

### 代码逐行解析

```python
def compute_ppl(model, loader, device):
    model.eval()                          # 切换到评估模式（关闭 Dropout、固定 BN）
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")  # 累加所有 token 的 loss，不求均值

    with torch.no_grad():                 # 不计算梯度，节省显存和计算
        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            logits = model(input_ids)              # (B, T, V)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B*T, V), targets.view(B*T))  # 展平后计算
            total_loss += loss.item()
            total_tokens += B * T                  # 累计 token 总数

    avg_loss = total_loss / total_tokens  # 所有 token 的平均 loss
    ppl = math.exp(avg_loss)              # PPL = e^(avg_loss)
    model.train()                         # 切回训练模式
    return ppl
```

### 为什么用 `reduction="sum"` 而非默认的 `"mean"`？

因为不同 batch 的 token 数可能不同（最后一个 batch 可能不满），先 sum 再手动除以总 token 数更精确。

---

## 五、train 函数 — 核心训练流程

### 5.1 设备选择

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

有 GPU 用 GPU，没有就用 CPU。没有硬编码设备。

### 5.2 数据加载

```python
train_path = DATA_DIR / f"train_seq{seq_len}.pt"   # data/train_seq256.pt
val_path = DATA_DIR / f"val_seq{seq_len}.pt"       # data/val_seq256.pt
if not train_path.exists():
    raise FileNotFoundError(f"未找到 {train_path}，请先运行 prepare_dataset.py")
```

如果数据文件不存在，直接报错提示用户先跑预处理。

### 5.3 模型构建

```python
model = build_model(vocab_size=vocab_size, seq_len=seq_len).to(device)
```

`build_model` 使用默认超参数（d_model=384, n_heads=6, n_layers=6, d_ff=1536）构建 MiniGPT，约 18.8M 参数。

### 5.4 优化器：AdamW + 差异化权重衰减

```python
# 分组：2D+ 的权重矩阵做衰减，1D 的 bias 和 LayerNorm 不衰减
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

optimizer = AdamW([
    {"params": decay_params, "weight_decay": 0.1},    # 权重矩阵：L2 正则
    {"params": no_decay_params, "weight_decay": 0.0},  # bias/LN：不衰减
], lr=3e-4, betas=(0.9, 0.95))
```

| 参数组 | 包含什么 | weight_decay | 原因 |
|---|---|---|---|
| decay_params | Linear.weight, Embedding.weight | 0.1 | 防过拟合，稀疏化权重 |
| no_decay_params | Linear.bias, LayerNorm.weight/bias | 0.0 | 偏置和归一化参数不需要正则化 |

> `betas=(0.9, 0.95)`：GPT 系列的标准设置，第二矩估计更保守（0.95 vs 默认 0.999），训练更稳定。

### 5.5 学习率调度：余弦退火

```python
total_steps = len(train_loader) * epochs    # 总训练步数
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)
```

学习率变化曲线：

```
lr
│ 3e-4 ───╮
│          ╲
│           ╲
│            ╲
│             ╲
│ 3e-5 ───────╯
└──────────────→ step
0          total_steps
```

- 前 50% 步数：lr 从 3e-4 平滑下降
- 到 `total_steps` 时：lr 降至 `3e-4 × 0.1 = 3e-5`（eta_min）
- **好处**: 前期快速学习，后期精细调整，避免在最优解附近震荡

### 5.6 基线 PPL

```python
baseline_ppl = compute_ppl(model, val_loader, device)
logger.info(f"基线 val PPL：{baseline_ppl:.1f}（随机猜测约等于 vocab_size={vocab_size}）")
```

训练前先测一次 PPL，此时模型随机初始化，PPL 应接近 21128。这是一个 sanity check。

### 5.7 训练循环（核心）

```
for epoch in 1..N:
    for batch in train_loader:
        ① 前向传播 → logits
        ② 计算损失 → loss
        ③ 清零梯度
        ④ 反向传播 → 计算梯度
        ⑤ 梯度裁剪 → 防止梯度爆炸
        ⑥ 更新参数 → optimizer.step()
        ⑦ 更新学习率 → scheduler.step()
    每 epoch 结束:
        ⑧ 验证集 PPL
        ⑨ 记录日志
        ⑩ 保存检查点
        ⑪ 更新最优模型
```

#### ①② 前向传播与损失计算

```python
logits = model(input_ids)              # (B, T, V) = (32, 256, 21128)
B, T, V = logits.shape
loss = loss_fn(logits.view(B*T, V), targets.view(B*T))
```

展平原因：`CrossEntropyLoss` 要求输入 `(N, C)` 和目标 `(N,)`。

```
logits:  (B, T, V) ──view──→ (B×T, V)     每行是一个位置的词表得分
targets: (B, T)    ──view──→ (B×T,)        每个元素是真实 token ID
```

#### ③④⑤⑥⑦ 梯度更新五步

```python
optimizer.zero_grad()                                           # ③ 清零上一步梯度
loss.backward()                                                 # ④ 反向传播，计算 ∂L/∂θ
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ⑤ 裁剪
optimizer.step()                                                # ⑥ 更新参数 θ
scheduler.step()                                                # ⑦ 调整学习率
```

#### Gradient Clipping 详解

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

- 计算所有参数梯度的 L2 范数：`||g|| = sqrt(Σ gi²)`
- 如果 `||g|| > 1.0`，将所有梯度缩放为 `g × (1.0 / ||g||)`
- 如果 `||g|| ≤ 1.0`，不做任何操作

> 为什么需要？Transformer 训练中偶尔会出现梯度突然放大（loss spike），裁剪防止一步更新太大导致模型崩溃。

#### ⑧ 验证集评估

```python
val_ppl = compute_ppl(model, val_loader, device)
train_ppl = math.exp(epoch_loss / n_batches)
```

每个 epoch 结束后在验证集上计算 PPL，与训练集 PPL 对比判断是否过拟合。

#### ⑨ 日志记录

```python
log_entry = {
    "epoch": epoch,
    "global_step": global_step,
    "train_loss": epoch_loss / n_batches,
    "train_ppl": train_ppl,
    "val_ppl": val_ppl,
    "lr": cur_lr,
}
with open(LOG_PATH, "a", encoding="utf-8") as f:
    f.write(json.dumps(log_entry) + "\n")
```

每行一个 JSON，便于后续用 `evaluate.py` 绘制训练曲线。示例：

```json
{"epoch": 1, "global_step": 4500, "train_loss": 7.2, "train_ppl": 1339, "val_ppl": 1420, "lr": 2.8e-4}
{"epoch": 2, "global_step": 9000, "train_loss": 5.8, "train_ppl": 330, "val_ppl": 380, "lr": 1.5e-4}
```

#### ⑩⑪ 检查点保存策略

```python
# 每个 epoch 都保存
ckpt_path = CKPT_DIR / f"epoch{epoch}_ppl{val_ppl:.1f}.pt"
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_ppl": val_ppl,
    "vocab_size": vocab_size,
    "seq_len": seq_len,
}, ckpt_path)

# 只保留最优模型
if val_ppl < best_val_ppl:
    best_val_ppl = val_ppl
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_ppl": val_ppl,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
    }, CKPT_DIR / "best_model.pt")
```

| 保存内容 | 说明 |
|---|---|
| `model_state_dict` | 模型权重，推理/恢复训练都需要 |
| `optimizer_state_dict` | 优化器动量等状态，恢复训练时需要 |
| `vocab_size`, `seq_len` | 模型配置，加载时必须匹配 |
| `val_ppl` | 评估指标，方便从文件名直接比较 |

> `best_model.pt` 不保存优化器状态（只用于推理），节省磁盘空间。

---

## 六、命令行参数

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
```

| 参数 | 默认值 | 说明 | 调优建议 |
|---|---|---|---|
| `--epochs` | 3 | 训练轮数 | 先跑 1 轮看效果，再增加 |
| `--batch_size` | 32 | 批大小 | 显存不足时减为 16 或 8 |
| `--lr` | 3e-4 | 峰值学习率 | 小模型通常 1e-4 ~ 5e-4 |
| `--weight_decay` | 0.1 | 权重衰减 | 防过拟合，0.05~0.2 常见 |
| `--grad_clip` | 1.0 | 梯度裁剪阈值 | 1.0 是 GPT 系列标准值 |
| `--seq_len` | 256 | 序列长度 | 需与预处理时一致 |
| `--num_workers` | 0 | 数据加载线程 | Windows 建议保持 0 |

---

## 七、完整训练流程图

```
prepare_dataset.py 的输出
        │
        ▼
  train_seq256.pt  val_seq256.pt
        │                │
        ▼                ▼
   TokenDataset     TokenDataset
        │                │
        ▼                ▼
   DataLoader        DataLoader
   (shuffle=True)   (shuffle=False)
        │                │
        ▼                │
  ┌──────────┐            │
  │  MiniGPT  │◄──────────┘（基线 PPL 评估）
  └──────────┘
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │             训练循环 (每个 epoch)              │
  │                                              │
  │  for batch in train_loader:                  │
  │    input_ids, targets ──model──→ logits      │
  │    logits + targets ──CrossEntropy──→ loss   │
  │    loss.backward()  → 计算梯度               │
  │    clip_grad_norm_() → 裁剪梯度               │
  │    optimizer.step()  → 更新参数               │
  │    scheduler.step()  → 调整学习率             │
  │                                              │
  │  epoch 结束:                                  │
  │    compute_ppl(val_loader) → val PPL          │
  │    保存 checkpoint + 更新 best_model.pt       │
  │    写入 training_log.jsonl                    │
  └─────────────────────────────────────────────┘
        │
        ▼
  outputs/
  ├── checkpoints/best_model.pt
  └── training_log.jsonl
```

---

## 八、单步训练的梯度流动

以一个 batch 为例，展示从输入到参数更新的完整数据流：

```
Step 1: 前向传播
  input_ids (32, 256)
        │
        ▼  MiniGPT.forward()
  logits (32, 256, 21128)
        │
        ▼  view + CrossEntropyLoss
  loss (标量)

Step 2: 反向传播
  loss
    │  .backward()
    ▼
  ∂L/∂θ 填入每个参数的 .grad 属性

Step 3: 梯度裁剪
  检查 ||grad|| 是否 > 1.0
    │  是 → 缩放所有梯度使 ||grad|| = 1.0
    │  否 → 不动
    ▼
  裁剪后的梯度

Step 4: 参数更新（AdamW）
  对每个参数 θ:
    m = β₁·m + (1-β₁)·grad          ← 一阶矩（动量）
    v = β₂·v + (1-β₂)·grad²         ← 二阶矩
    m̂ = m / (1-β₁^t)                ← 偏差修正
    v̂ = v / (1-β₂^t)
    θ = θ - lr · m̂ / (√v̂ + ε) - lr · weight_decay · θ  ← 解耦权重衰减

Step 5: 学习率更新
  lr = eta_min + 0.5 · (lr_init - eta_min) · (1 + cos(π · step / T_max))
```

---

## 九、训练过程中各指标的预期变化

| 阶段 | train_loss | train_PPL | val_PPL | lr | 说明 |
|---|---|---|---|---|---|
| 论练前 | ~9.96 | ~21000 | ~21000 | 3e-4 | 接近 ln(21128)，随机猜测水平 |
| Epoch 1 结束 | ~7.0 | ~1100 | ~1300 | ~2.8e-4 | 快速下降，模型开始学到词频 |
| Epoch 2 结束 | ~5.5 | ~245 | ~320 | ~1.5e-4 | 学到局部搭配 |
| Epoch 3 结束 | ~4.8 | ~120 | ~180 | ~3e-5 | 学到更长范围的依赖 |

> 以上为参考值，实际结果取决于语料量、模型大小和训练配置。

---

## 十、关键设计决策总结

| 设计 | 选择 | 原因 |
|---|---|---|
| 损失函数 | CrossEntropyLoss | 自回归 LM 标配，等价于最大似然估计 |
| 优化器 | AdamW | 解耦权重衰减，比 Adam 更稳定 |
| 权重衰减策略 | 差异化（2D+ 衰减，1D 不衰减） | bias 和 LN 不需要正则，否则限制表达能力 |
| LR 调度 | CosineAnnealingLR | 平滑衰减，末期精细收敛 |
| betas | (0.9, 0.95) | GPT 标准配置，比默认 (0.9, 0.999) 更保守 |
| 梯度裁剪 | clip_grad_norm_=1.0 | 防梯度爆炸，1.0 是 GPT 系列经验值 |
| 保存策略 | 每 epoch 保存 + best_model | 兼顾容错和磁盘空间 |
| PPL 评估 | reduction=sum + 手动均值 | 跨 batch 精确计算，不受 batch 大小影响 |
