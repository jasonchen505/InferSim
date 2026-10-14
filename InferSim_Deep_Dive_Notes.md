# InferSim 深入学习文档

> 定位：个人学习笔记，只放在 fork 的 `notes/learning` 分支，不进上游 PR，不动 `main`。
> 基线：本文基于上游 `alibaba/InferSim` HEAD `d02a40c`（2026-09-04，Add DeepSeek-V4 H20 kernel benchmark data #17）通读全部核心代码并实测两条 example 后写成。
> Fork 现状：`jasonchen505/InferSim` 的 `main` 仍停在 `cbbc22b`（2026-01-04，fix code style），落后上游 13 个 commit（见第 9 节）。本分支就是从该旧 `main` 拉出的，文档内容描述的是上游最新版，不是 fork `main` 的旧代码。做任何上游 PR 之前必须先把 fork 同步到上游 `main`，再从上游 `main` 拉贡献分支。

## 1. 一句话定位

InferSim 是一个纯 Python、无第三方依赖的轻量 LLM 推理性能模拟器。输入是一份 HF `config.json` 加部署参数（GPU 型号、world size、TP size、prefill token 数、目标 TGS/TPOT 等），输出是 TTFT、TPOT、TGS（tokens/GPU/second），以及权重、KV Cache、FLOPs 的分项账本。

它不跑真实推理，不加载权重，不做算子执行。它做的是解析式建模：

```
latency ≈ FLOPs / (FLOPS × MFU)  与  显存搬运时间  取 max
通信时间 ≈ 数据量 / 带宽
TTFT / TPOT = 每层延迟求和 × 层数 + 调度开销
```

两个宣称用途：

1. **Model-Sys co-design**：给定模型超参，预测推理性能，指导模型结构与部署配置的联合设计。
2. **推理性能分析**：量化瓶颈是算力 bound 还是访存 bound，支撑优化方向判断。

作者：阿里巴巴 Alimama AI Infra Team 与 Future Living Lab。Apache-2.0。2025-09-28 创建。

## 2. 仓库现状（2026-10-14 核实）

| 指标 | 数值 |
|---|---|
| Stars / Forks | 103 / 30 |
| 上游最后 push | 2026-09-04（`d02a40c`） |
| Contributors | 7 人，主力 MtFitzRoy 44 commits，其次 sheshuchen 4 |
| Open issues | 5 个（#1、#3、#6、#7、#8），见第 8 节 |
| Open PR | 0。历史 PR 12 个，全部 closed，其中 8 个 merged，4 个未 merge 关闭（#9、#13、#15、#16，多为 DeepSeek-V4 适配的规划/中间 PR） |
| CI | 无 `.github/`，无 GitHub Actions。质量门只有 `.pre-commit-config.yaml`（ruff + isort + black） |
| 测试 | 只有 `tests/test_attn.py` 一个文件，3 个测试，全部针对 MLA decode（随 #19 加入） |
| 语言 | 纯 Python。核心模拟器只用标准库（argparse、json、csv、math、dataclasses）。`kernel_benchmark/` 需要 torch / FlashInfer / DeepGEMM / SGLang 等，只在有 GPU 的机器上跑 |

规模：核心模拟器（含 main、config、flops、hardware、kvcache、layers、mfu、models、params、comm、tests、kernel_sim）约 2467 行；`kernel_benchmark/` 约 4836 行；`bench_data/` 45 个 CSV。

评价：小而精属实。核心链路一上午能读完。但“冷”也属实：上游 2026-09-04 后无新 push，issue #6 从 2026-03-25 提出、04-09 获维护者确认，至今仍 open 无修复。这决定了它的贡献节奏：适合小而确定的修复，不适合等 maintainer 快速反馈的大 feature。

## 3. 总体架构与数据流

```
main.py (argparse)
  └─ ModelConfig (config/model_config.py, 解析 HF config.json)
  └─ Model / HybridModel (models/model.py, models/hybrid_model.py)
       ├─ print_weights_info()   → params/params.py
       ├─ print_kvcache_info()   → kvcache/kvcache.py
       ├─ print_flops_info()     → flops/flops.py
       ├─ prefill()  → layers/attn.py + layers/moe.py (+ layers/linear_attn.py) + comm/comm.py + mfu/mfu.py
       └─ decoding() → 同上
  bench_data/*.csv ← mfu/mfu.py 查表得到 MFU / 实测延迟
  hardware/gpu.py   → GPU 峰值算力、带宽、默认 MFU
```

关键分流在 `main.py`：

- `config.is_hybrid_linear` 为真（HF config 里有 `full_attention_interval`）→ `HybridModel`，否则 `Model`。
- `config.attn_type`：HF config 里有 `kv_lora_rank` → `MLA`，否则 `MHA/GQA`。
- `config.is_moe`：有 `num_routed_experts` 或 `num_experts` → MoE，否则 dense FFN 按“1 专家 MoE”处理。

文件地图：

| 目录/文件 | 职责 |
|---|---|
| `main.py` | CLI 入口，参数定义与校验（MFU 范围、调度开销非负） |
| `config/model_config.py` | HF config 解析，派生 attn 类型、混合层数、MoE 形状 |
| `flops/flops.py` | 纯 FLOPs 公式：MHA、MLA（absorb / noabsorb / GQLA）、MoE |
| `layers/attn.py` | Attention 层延迟：MHA、MLA，prefill/decode 各一对方法 |
| `layers/moe.py` | MoE/FFN 层延迟：routed + shared，支持 shared expert overlap |
| `layers/linear_attn.py` | 线性注意力 GDN 层延迟（Qwen3-Next / Qwen3.5 混合模型用） |
| `mfu/mfu.py` | 查 `bench_data` 得到 MFU 或实测延迟；查不到时 fallback 默认 MFU |
| `hardware/gpu.py` | 5 种 GPU 的峰值参数表 |
| `comm/comm.py` | 通信：all_reduce、DeepEP dispatch/combine、TP all_reduce |
| `params/params.py` | 参数量、权重加载时间、期望激活专家数 |
| `kvcache/kvcache.py` | 每 token KV Cache 大小、线性层 state 大小 |
| `models/model.py` | 普通模型：权重/KV/FLOPs 打印 + prefill + decode |
| `models/hybrid_model.py` | 混合模型：full attn 层 + linear attn 层加权求和 |
| `kernel_sim/sparse_mla_fp8.py` | 独立工具：稀疏 MLA FP8 的解析估计，不接入 main 链路 |
| `kernel_benchmark/` | 在真 GPU 上跑 FlashInfer / FA3 / DeepGEMM / SGLang，产出 `bench_data` |
| `bench_data/` | 实测 CSV：mha、mla、gemm、grouped_gemm、gdn、dsa |
| `hf_configs/` | 8 份示例 HF config：DeepSeek-V3/V3.2、Qwen3 系列、Qwen3-Next、Qwen3.5 |
| `example/` | 8 个可直接跑的 shell 脚本 |

## 4. 核心建模：四条公式就是全部

### 4.1 GEMM 延迟

```
latency = FLOPs / (FLOPS × MFU)
FLOPs(GEMM) = 2 × M × N × K
```

`flops/flops.py:gemm_flops(m, n, k) = 2*m*n*k`。`layers/attn.py:get_gemm_mfu_and_latency()` 查 MFU 后套用上式。`--use-fp8-gemm` 时分母换成 FP8 峰值算力，但 MFU 仍沿用同一张表（见第 7 节的待核实点）。

### 4.2 Attention core：算力与访存取 max

Decode（`MHA.decode_attn_core`、`MLA.decode_attn_core`）：

```
attn_core_time = bs × per_token_attn_core_GFLOPs / (FLOPS × MFU)   # 或实测延迟，见 5.5
kv_load_time   = kvcache_bytes × kv_len × bs / num_layers / 1GiB / mem_bw
return max(attn_core_time, kv_load_time)
```

例外：`MLA` 在 #19 之后，若 `bench_data` 有该 batch 与 kv_len 的实测点（或插值点），直接返回实测延迟，不再与 KV 加载取 max，理由是实测延迟已包含 KV 读取。MHA 的 decode 在 #19 之后也走了同一分支（见 `layers/attn.py`）。

Prefill（`MHA.prefill_attn_core`、`MLA.prefill_attn_core`）：

```
attn_core_time = seq_len × per_token_attn_core_GFLOPs / 1.8 / (FLOPS × MFU)
kv_load_time   = kvcache_bytes × seq_len / num_layers / 1GiB / mem_bw
return max(attn_core_time, kv_load_time)
```

`/ 1.8` 是 causal mask 的等效折扣，对应技术报告里的 `R_mask`。理论 causal 平均应为 `/ 2`，代码用 `/ 1.8`（即 `R_mask ≈ 0.556`），无注释说明来源，是待核实的经验系数。

### 4.3 MoE：计算与权重加载取 max，shared 可串可并

`layers/moe.py:MoE.decode_moe / prefill_moe`：

```
routed_gflops = bs × num_experts_per_tok × 3 × 2 × hidden × (intermediate / TP)   # 3 = gate + up + down
routed_latency = routed_gflops / (FLOPS × grouped_GEMM_MFU)
moe_load_time  = 期望激活专家数 × 每专家参数量 / mem_bw
t = max(routed_latency, moe_load_time)
if shared_experts > 0:
    shared_latency = up + down 两段 GEMM
    overlap 时 t = max(t, shared_latency)，否则 t += shared_latency
```

期望激活专家数（`params/params.py:get_expected_active_experts`）假设均匀路由：

```
P(某本地专家未被选中) = (1 - topk / num_local_experts) ^ num_tokens
E[active] = num_local_experts × (1 - P)
```

`num_tokens=None` 时（prefill 的权重加载路径）直接返回全部本地专家数，即假设 prefill 会触达所有专家。

### 4.4 通信：数据量 / 带宽

`comm/comm.py:Comm.size_bw_model`：`size(GB) / bw`。`world_size <= 1` 时返回 0。

- 普通路径：`prefill_comm / decode_comm` 在未开 DeepEP 时都返回两次 `all_reduce`（MoE 前后各一次）。
- DeepEP：prefill 用 normal 模式 dispatch/combine，decode 用 low_latency 模式。normal 模式的跨节点量按 `(num_nodes - 1)` 缩放，low_latency 按 `num_experts_per_tok` 缩放。
- TP：`tp_all_reduce` 用 ring all_reduce 公式，传输量 `2 × (TP-1)/TP × tokens × hidden × 2 字节`，走 NVLink 带宽。TP=1 时返回 0。

### 4.5 TTFT / TPOT / TGS 的组装

`Model.prefill`（`models/model.py`）：

```
ttft = (attn_core + attn_other + moe + comm1 + comm2 + tp_comm) × num_layers
ttft = ttft × 1000 (s → ms) + 30  # 30ms 固定调度开销
TGS  = max_prefill_tokens / TP / (ttft / 1000)
```

其中 `attn_core_time` 先按 `target_isl` 算，再乘 `ceil(max_prefill_tokens / target_isl)`。`HybridModel.prefill` 用的是浮点比 `max_prefill_tokens / target_isl`，两者在不可整除时不一致（见第 7 节）。

`Model.decoding`：

```
tpot = (attn_core + attn_other + moe + comm1 + comm2 + tp_comm) × num_layers
tpot = tpot × 1000 + scheduler_overhead  # 默认 5ms，HybridModel 默认 2ms
TGS  = decode_bs / TP / (tpot / 1000)
```

TBO（`--enable-tbo`，two batch overlap）时，prefill/decode 改用 `max(计算, 通信)` 的重叠求和，并除以 `sm_ratio`（默认 `108/132`，即 H800 的 SM 数比）。非 TBO 时是纯串行相加。

`decode_bs` 未指定时：`target_bs = ceil(target_tgs × target_tpot / 1000)`，即 Little 定律的 batch 估计。`avg_context_len = ISL + OSL/2`，即假设 decode 平均已生成一半输出。

## 5. 各模块细读

### 5.1 ModelConfig（`config/model_config.py`）

- Qwen3.5 特殊处理：`model_type` 为 `qwen3_5` / `qwen3_5_moe` 时，取 `text_config` 子字典作为真实配置，并据此判定 `is_qwen3_5_moe`。
- 混合判定：`full_attention_interval` 存在 → 混合模型。`num_full_attn_layers = num_hidden_layers // full_attention_interval`，其余为线性层。注意是整除，未校验整除性。
- MLA 判定：仅看 `kv_lora_rank` 是否在字典里。`index_topk` 用 `.get`，缺失为 None（DSA 稀疏用，见第 8 节 #3）。
- MoE 判定：`num_routed_experts` 优先，否则 `num_experts`，都没有则 dense，`num_routed_experts=1`。
- Shared expert 约束：若 `shared_expert_intermediate_size` 不是 `moe_intermediate_size` 的整数倍，直接抛 `ValueError`。这是全仓少见的显式校验。
- 未做校验的：`head_dim` 缺失时默认 `hidden // num_heads`；TP 整除性（`num_heads // tp_size` 等）全程用整除，未校验余数。

### 5.2 FLOPs（`flops/flops.py`）

| 函数 | 公式（per token，bs=1） |
|---|---|
| `get_mha_gflops` | Q/K/V/O 四段 GEMM + attn core。Q 的输出维度乘 `q_multiplier`（`attn_output_gate` 时为 2，即 Qwen3.5 的 gated attention）。attn core = `2·bs·D·S + 2·bs·S·D`，其中 `D = tp_heads × head_dim`，`S = avg_context_len` |
| `get_mla_absorb_gflops` | Q down/up、KV down、吸收后的 `bmm_q_wk` / `bmm_o_wv`、O。attn core 用 `kv_lora_rank + rope` 做 QK 维度，用 `kv_lora_rank` 做 AV 维度 |
| `get_mla_noabsorb_gflops` | 多一段 KV up（`kv_lora → heads×(v_head + qk_nope)`）。attn core 用完整 `qk_nope + rope` 与 `v_head_dim` |
| `get_gqla_absorb_gflops` | GQLA 变体，把 kv_lora 对半拆。注意函数内 `2 × heads / 2` 实际等于 heads，写法冗余但结果正确。且该函数没有 `tp_size` 参数，TP 下不分头 |
| `get_moe_gflops` | `(shared + topk) × 3 × 2 × hidden × (intermediate / TP)` |

TP 规则统一写在注释里：TP 只切 head 数 / intermediate_size，不切 `hidden_size`。这与 Megatron 的列并行/行并行 + allreduce 一致。

### 5.3 Attention 层（`layers/attn.py`）

- `MHA.get_attn_core_gflops(bs, kv_len)`：`gemm_flops(bs, D, kv_len) × 2`，即 QK 与 AV 两项，`gemm_flops` 自带的 2 已计入，`×2` 是两项求和。
- `MHA.decode_attn_others`：QKV 合并成一个 GEMM（`m=bs, k=hidden, n=(q_mult×tp_heads + 2×tp_kv_heads)×head_dim`），O 单独一个 GEMM。Prefill 的 others 直接复用 decode 版本，传入 `m=max_prefill_tokens`。
- `MLA.decode_attn_others`：拆成 6 段 GEMM 逐个查 MFU 求和（Q down、Q up、KV down、`bmm_q_wk`、`bmm_o_wv`、O）。`bmm_q_wk` / `bmm_o_wv` 的形状在代码里写成了 GEMM 形状（`m=bs`），实际是 batched matmul，建模上按 GEMM 近似。
- `MLA.prefill_attn_core` 用 noabsorb 的 core，`MLA.decode_attn_core` 用 absorb 的 core，`print_flops_info` 用 absorb。三者口径不一致，见第 7 节。
- `create_attention` 是简单工厂，按 `attn_type` 返回 MHA 或 MLA。MLA 的 `__init__` 不接收 `prefill_mfu`，所以 `--prefill-attn-mfu` 对 MLA prefill 无效（MHA 才用）。这是参数覆盖范围的一个缺口。

### 5.4 MoE 层（`layers/moe.py`）

- Dense FFN 按 1 专家 MoE 处理：`routed_experts_mfu` 走普通 GEMM 查表，`intermediate × 2 // num_gpus` 是 `intermediate` 拆到各 GPU 的近似。
- MoE 的 MFU 取 `max(up_mfu, down_mfu)`（`get_groupedgemm_*` 返回一对，取 max）。逻辑上是两段 GEMM 串行，取 max 会低估总延迟，但代码如此。
- Shared expert 的 up 段：`n = tp_intermediate × 2 × num_shared`（`×2` 是 gate+up 合并），down 段 `k = tp_intermediate × num_shared`。`decode_moe` 支持 overlap，`prefill_moe` 永远串行相加。
- `prefill_moe` 的权重加载不传 `num_tokens`，即按全部本地专家计，与 decode 的“期望激活数”口径不同。

### 5.5 MFU 查表（`mfu/mfu.py`）—— 精度的命脉

README 明确说：模拟精度高度依赖 kernel benchmark 结果。这是全仓最重要的一句话。

| 函数 | 查表策略 | 查不到时 |
|---|---|---|
| `get_attn_decode_perf` | 文件名按 `tp_heads-tp_kv_heads-head_dim` / MLA 按 `tp_heads-kv_lora-rope` 定位。同一 batch 下：kv_len 精确命中返回实测延迟；介于两点之间线性插值；超出范围取端点 MFU 但延迟返回 None（走解析式）。batch 不命中时取最近 batch 的最近 kv_len 点的 MFU，延迟 None | 文件不存在 → 警告 + 默认 MFU，延迟 None。无该 dtype 行 → 同样 fallback |
| `get_attn_prefill_mfu` | 按 seq_len 阶梯查：取最后一个 `≤ seq_len` 的行的 MFU | 文件不存在 → 0.9。注意与 GPU 默认 MFU（0.35–0.6）不同 |
| `get_groupedgemm_decode_mfu` | 精确匹配专家数、GPU 数、本地专家数、topk、hidden、intermediate 后，取 batch 最近行的 up/down MFU | 文件不存在或无匹配行 → 默认 MFU |
| `get_groupedgemm_prefill_mfu` | 同上，阶梯查 seq_len | 同上 |
| `get_gemm_mfu` | 在 `k ≥ K 且 n ≥ N` 的行中找 `(k-K)² + (n-N)²` 最小的 `(k,n)`，再在该组里取 `m ≤ M` 的最大 m 行的 MFU。注意 `m` 不参与最近邻，只作上限过滤 | 无匹配 → 默认 MFU + 警告 |
| `get_linear_attn_*_latency`（GDN） | 取最近 seq_len / batch 的行，各列求和后按 `row / target` 比例缩放 | 文件不存在 → `assert False`，直接崩溃。与其他函数的 fallback 风格不一致 |

统一规律：decode 优先实测延迟，prefill 只用 MFU 做解析缩放。查表全部用相对路径 `bench_data/...`，必须在仓库根目录运行，否则一律 fallback（只打印 Warning）。

### 5.6 GPU 表（`hardware/gpu.py`）

| GPU | FP16 TFLOPS | FP8 TFLOPS | 默认 MFU | 显存 GB | 显存带宽 GB/s（已打折） | NVLink 单向 GB/s | RDMA 单向 GB/s |
|---|---|---|---|---|---|---|---|
| H20 | 148 | 296 | 0.60 | 96 | 4096×0.8 | 900×0.8/2 | 50×0.8 |
| H800 | 989 | 1979 | 0.35 | 80 | 3430×0.8 | 400×0.8/2 | 50×0.8 |
| H200 | 989 | 1979 | 0.40 | 141 | 4800×0.8 | 900×0.8/2 | 50×0.8 |
| GB200 | 2500 | 5000 | 0.50 | 192 | 13400×0.8 | 1800×0.8/2 | 50×0.8 |
| PRO5000 | 274 | 536 | 0.50 | 72 | 1345×0.85 | 64×0.8 | 50×0.8 |

注释说 `1 TFLOP/s = 1000 GFLOP/s`（`TFLOPS_TO_GFLOPS = 1000`）。带宽与 NVLink 普遍乘 0.8 效率折扣，PRO5000 显存带宽乘 0.85，`frequency` 乘 0.9。H800/H200 的 `frequency`/`num_sm`/`sm_version` 不全，PRO5000 的 `frequency` 为 None（`kernel_sim` 用到 `frequency` 时会崩）。

### 5.7 通信（`comm/comm.py`）

见 4.4。补充两个细节：

- `all_reduce` 的张量形状是 `[num_tokens × world_size, hidden]`，即假设每个 GPU 的 token 数为 `num_tokens`，all_reduce 前先拼接。`dispatch/combine` 的 normal 模式按 `num_nodes` 缩放，隐含假设每个节点一份数据。
- `size_bw_model` 中 `use_fp8=True` 时按 1 字节计，否则 2 字节。`combine` 的 normal 模式用 FP16（2 字节），`dispatch` 用 FP8（1 字节），不对称是 DeepEP 的约定（dispatch 量化、combine 不量化）。

### 5.8 参数与 KV（`params/params.py`、`kvcache/kvcache.py`）

- `get_mha_params_size`：Q/K/V/O 四矩阵，FP8 时 1 字节，否则 2 字节。Q 带 `attn_output_gate` 时 `×2`。
- `get_mla_params_size`：Q down/up、KV down/up、O。见第 7 节待核实点：KV down 只计 `hidden × kv_lora_rank`，未计 rope 部分，而 FLOPs 里的 KV down 计了 `kv_lora_rank + rope`。
- `get_gdn_params_size`：Q/K/V/Z/A/B 六段 + conv。conv 里 key 的卷积计了两次（Q 与 K 各一），value 计一次。FP8 时 `s + wconv`，非 FP8 时 `2×s + wconv`，即 conv 权重恒按 1 字节计（疑似约定，未注释）。
- `get_expert_params_size`：`3 × hidden × (intermediate / TP)`，FP8 时 1 字节，否则 2 字节。
- KV：MHA 每 token `2 × num_layers × tp_kv_heads × head_dim`（×2 是 K+V，再 ×2 是 bf16 字节数）；MLA 每 token `num_layers × (kv_lora_rank + rope)`（×2 是 bf16）。MLA 的 KV 不随 TP 切分（注释明确）。
- GDN state：`conv_state + ssm_state`，conv 按 bf16（×2），ssm 按 fp32（×4）。
- `print_weights_info` / `print_kvcache_info` 里的显存账本：`kvcache_mem = gpu.mem - params_per_gpu - 15 - 5`，15GB 运行时保留、5GB encoder 保留，硬编码，无配置项。

### 5.9 HybridModel（`models/hybrid_model.py`）

与 `Model` 的差异：

- 权重：full attn 与 linear attn 分开计，`params_per_gpu = expert×层数 + full×full层数 + linear×linear层数`。
- KV：`kvcache_bytes` 只计 full attn 层（`get_kvcache_size / num_layers × num_full × context_len`），再加 `states_bytes`（全部 linear 层的 state）。`self.kvcache_bytes` 存的是“每 token 的 full attn KV”（除以 context_len 后），供 attn core 的访存项使用。
- Prefill/decode：`full` 与 `linear` 两部分分别算延迟，再按 `num_full` / `num_linear` 加权。TP>1 时通信用 `2 × tp_comm × 层数`，TP=1 时用 `(comm1+comm2) × 层数`。注意 `Model` 在 TP>1 时是 `comm1+comm2` 不计（因 `prefill_comm` 在 TP>1 时仍返回 all_reduce？实际 `Model.prefill` 里 TP>1 时仍把 `comm1+comm2` 排除，只加 `tp_comm`；`HybridModel` 在 TP>1 时加 `2×tp_comm`，即每层两次 TP all_reduce（attn 后一次、MoE 后一次），与 `Model` 的一次不同。见第 7 节。
- 解码调度开销默认 2ms，`Model` 默认 5ms。Prefill 固定 +30ms，两者相同。

### 5.10 kernel_sim 与 kernel_benchmark

- `kernel_sim/sparse_mla_fp8.py`：独立 CLI，估计 DeepSeek 稀疏 MLA FP8 的时间与理论 TFLOPS。建模粒度到 block（`BLOCK_M=64`）、反量化周期（`DEQUANT_CYCLES` 只定义了 SM90）、KV 加载与反量化取 max。不接入 `main.py`，是第 8 节 #3 讨论的“DSA 已有零散实现、未进主链路”的实证。
- `kernel_benchmark/`：22 个脚本，覆盖 DeepGEMM（GEMM / grouped GEMM / FP8 logits）、FlashInfer（MHA/MLA decode/prefill、GEMM）、FA3（MHA/MLA prefill）、FlashMLA（DSA decode/prefill、DeepSeek-V4 稀疏 prefill）、SGLang（causal conv1d、GDN、fused MoE、grouped GEMM、topk）、显存带宽。每个脚本自带 sweep 参数与正确性校验（部分带 `assert` / `check_is_allclose`）。`run_all_benchmarks.sh` 是批量入口。

## 6. 精度：README 表格怎么读

| Model | GPU | Prefill TGS 实际 / 模拟 | 误差 | Decode TGS 实际 / 模拟 | 误差 |
|---|---|---|---|---|---|
| DeepSeek-V3 | H800 | 7839 / 9034 | +15.3% | 2324 / 2675 | +15.1% |
| Qwen3-30B-A3B-BF16 | H20 | 16594 / 17350 | +4.6% | 2749 / 2632 | -4.3% |
| Qwen3-8B-FP8 | H20 | 15061 / 16328 | +8.4% | 2682 / 2581 | -3.8% |
| Qwen3.5-35B-A3B-FP8 | PRO5000 | 33139 / 36449 | +10.0% | 2150 / 2175 | +1.2% |

读法：

- 误差在 ±15% 内，且多数在 ±10% 内。对解析式模拟器而言是可用水平，但不是“准”。
- DeepSeek-V3 的两项都高估 15%，方向一致，暗示 MLA 路径的某个系统性乐观假设（MFU 查表、通信、或吸收近似）。
- Qwen3 系列的 prefill 高估、decode 低估，方向不一致，说明 prefill 与 decode 的误差来源不同（prefill 的 `/1.8`、decode 的实测延迟插值）。
- README 自己说：精度高度依赖 kernel benchmark 结果。这是诚实且关键的限定。

实测复现（本机，无 GPU，纯解析模拟）：

```
Qwen3-30B-A3B / H20 / world=4 / decode-bs=100 / decode-only
  TPOT 32.27ms，TGS 3099（README 例为 38.00ms / 2632，因 README 例的 target_tgs/tpot 与当前默认参数不同，属参数差异非代码漂移）
Qwen3-30B-A3B / H20 / world=1 / max-prefill=16384 / prefill-only
  TTFT 971.79ms，TGS 16860，并打印 "!Error: need smaller kvcache"（因单卡 KV 空间不足，见第 7 节）
```

## 7. 假设、魔法数与待核实点

以下按“已确认的事实”与“待核实的观察”分开。观察不是 bug 断言，需与 maintainer 或实测对照后才可提 PR。

### 已确认的建模假设

- 无队列、无调度器模拟。`decode_bs` 由 Little 定律反推，不是模拟排队得到。
- 无 chunked prefill、无 PD 分离、无前缀缓存、无投机解码。Prefill 与 decode 各自独立算一次，不模拟混合 batch。
- 通信与计算除 TBO 外全部串行相加。真实引擎的 overlap 只在 TBO 分支里近似。
- FP8 只换分母峰值算力（GEMM）与 KV 字节数，不换 MFU 表。`get_gemm_mfu_and_latency` 在 FP8 时仍用同一 MFU。
- 所有 `bench_data` 查表用相对路径，必须在仓库根目录运行。

### 魔法数清单

| 位置 | 数值 | 含义 / 备注 |
|---|---|---|
| `layers/attn.py` prefill | `/ 1.8` | causal mask 等效折扣，约 `R_mask=0.556`，无来源注释 |
| `models/*.py` | `- 15 - 5` | 显存保留：15GB 运行时 + 5GB encoder，硬编码 |
| `models/model.py` prefill | `+ 30` ms | 固定调度开销 |
| `models/model.py` decode | 默认 5ms | 调度开销，可用 `--decode-scheduler-overhead-ms` 覆盖 |
| `models/hybrid_model.py` decode | 默认 2ms | 与 Model 的 5ms 不一致，无注释 |
| `main.py` | `sm_ratio = 108/132` | TBO 时计算侧 SM 占比，108 是 H800 的可用 SM？132 是总数。其他 GPU 下仍用此默认值 |
| `hardware/gpu.py` | `×0.8` / `×0.85` / `×0.9` | 带宽、NVLink、频率的效率折扣，无来源 |
| `mfu/mfu.py` | prefill 查不到时 0.9 | 与 GPU 默认 MFU（0.35–0.6）量级不同 |
| `models/model.py` | `avg_context_len = ISL + OSL/2` | decode 平均上下文假设 |

### 待核实的观察（不是结论）

1. **HybridModel 的 FLOPs 打印口径**：`print_flops_info` 里 `Per-token total = (attn_core + moe + others) × num_hidden_layers`，其中 `attn_core` 是 full attn 的值，却乘了全部层数；linear attn 的 FLOPs 未计入。`prefill` / `decoding` 里是按 `num_full` / `num_linear` 加权的，与打印口径不一致。
2. **MLA 三处口径**：prefill 用 noabsorb core，decode 用 absorb core，`print_flops_info` 用 absorb。若三者是刻意区分（prefill 不吸收、decode 吸收），应在注释中说明；当前无注释。
3. **MLA `--prefill-attn-mfu` 无效**：`MLA.__init__` 不接收 `prefill_mfu`，`create_attention` 传给 MLA 的 `prefill_mfu` 被丢弃。MHA 才生效。
4. **MLA 参数量 vs FLOPs 的 rope 口径**：`get_mla_params_size` 的 KV down 只计 `hidden × kv_lora_rank`，`get_mla_absorb_gflops` 的 KV down 计 `hidden × (kv_lora_rank + rope)`。以 DeepSeek-V3（hidden 7168，rope 64，61 层）计，差约 28M 参数（约 56MB bf16），相对总量可忽略，但口径不一致。
5. **MoE MFU 取 max**：`decode_moe` / `prefill_moe` 对 grouped GEMM 的 up/down MFU 取 `max`，而两段是串行。取 max 会低估延迟。
6. **TP 通信计数**：`Model` 在 TP>1 时只加一次 `tp_comm`，`HybridModel` 加 `2 × tp_comm × 层数`。两者对“每层几次 TP all_reduce”的假设不同，无注释。
7. **Prefill 的 token 数缩放**：`Model` 用 `ceil(max / target_isl)`，`HybridModel` 用浮点比。不可整除时 `Model` 高估。
8. **GDN 查表失败直接崩**：`get_linear_attn_*_latency` 文件不存在时 `assert False`，与其他 MFU 函数的 fallback 风格不一致。
9. **GDN conv 权重字节**：`get_gdn_params_size` 里 `wconv` 恒按 1 字节计，即使非 FP8 时 `s` 按 2 字节计。可能是约定，未注释。
10. **`get_gqla_absorb_gflops` 无 TP 参数**：TP 下不分头，与其他 MLA 函数不一致；且 `2 × heads / 2` 的写法冗余。

## 8. Issue 与 PR 全景

### Open issues（5 个）

| # | 标题 | 状态 | 要点 |
|---|---|---|---|
| #1 | InferSim tech report | open，2025-09-28 | 维护者贴了技术报告 PDF 链接（`infersim_tech_report.pdf`）。报告不在仓库内，只有附件链接 |
| #3 | Deepseek v3.2 no supported? | open，2026-01-05 | 主链路未实现 DSA（DeepSeek-V3.2 的稀疏注意力）。`index_topk` 已在 config 里，但缺 `index_head_dim` 等字段；`kernel_sim` 与部分 CSV 已有 DSA 零散实现，未接入 main。维护者 2026-02-10 回复：KV Cache offload 模拟已有结果，但未准备好开源。后续追问（2026-03-25）无进一步回复 |
| #6 | FLOPs of Q, K, V, and O projections | open，2026-03-25 | 见下文专项。维护者 2026-04-09 已确认正确，但至今无修复 |
| #7 | Future plans and the project general idea | open，2026-03-30 | 提议：单卡多模型（KV 不占满全部剩余显存）、更多 GPU（点名 A100）。维护者 2026-04-09 回复：项目聚焦 LLM、需要一卡或多卡；可模拟时分多模型服务（权重与 KV offload 到 CPU）；欢迎贡献，包括更多 GPU 类型 |
| #8 | HiSim vs InferSim | open，2026-04-29 | 问同为阿里系的 hisim（tair-cache）与 InferSim 的区别。维护者 2026-05-07 回复：不同团队；InferSim 是轻量模拟器，HiSim 功能更详细 |

已关闭的 #4（`--max-prefill-tokens` 与 `--decode-bs` 是单卡还是总量）：维护者确认是单卡，并承诺补充 help 信息。后续 `--help` 文案已改为“per GPU”。

### #6 专项：技术报告公式漏 `S_q`，代码是对的

Issue #6 贴的是技术报告第 3 页截图，红箭头指第二行公式：

```
报告原文：FLOPs_qkvo_proj = 4 (n_h + n_kv) d_h d_hidden
应为：    FLOPs_qkvo_proj = 4 S_q (n_h + n_kv) d_h d_hidden
```

推导：Q、K、V、O 各是一次 GEMM（`M=S_q, N=输出维度, K=d_hidden`），每段 `2 × S_q × d_hidden × d_out`。Q 与 O 的 `d_out = n_h × d_h`，K 与 V 的 `d_out = n_kv × d_h`，四段求和即 `2 × S_q × d_hidden × (2n_h d_h + 2n_kv d_h) = 4 S_q (n_h + n_kv) d_h d_hidden`。报告漏了 `S_q`，而同一页第一行 attn core 公式 `4 n_h S_q S_kv d_h R_mask` 是带 `S_q` 的，对照即显。

代码侧已正确：`flops/flops.py:get_mha_gflops` 的 Q/K/V/O 均以 `bs`（即 `S_q`）为 `m`；`layers/attn.py:decode_attn_others` 以 `bs` 为 GEMM 的 `m`；prefill 时传入 `m=max_prefill_tokens`。因此 #6 的修复不在代码，而在技术报告 PDF 本身（附件，不在仓库内）或在仓库内补充公式注释/回归测试。若直接给代码再乘一次 `S_q`，会重复计算。

维护者 MtFitzRoy 2026-04-09 回复：“Yes, you are right. Thanks for your correction.”此后无后续 commit。按本仓的贡献经验，这类“已确认但未修”的文档类 issue 是最小的切入点，但需先确认 maintainer 接受何种形式的修复（重发 PDF、仓库内公式说明、还是代码注释 + 测试），否则 PR 可能无处落。

### PR 历史（12 个，8 merged）

- #2（2025-12-25 merged）：稀疏 MLA 性能估计模块，即 `kernel_sim/sparse_mla_fp8.py`。
- #5（2026-03-18 merged）：适配 Qwen3.5-35B-A3B-FP8。
- #10/#11（2026-05-07/08 merged）：Qwen3.5-122B-A10B 的 bench data + HF config + TP 支持与模型更新。
- #12/#14（2026-07-16 merged）：Qwen3.5-35B-A3B-FP8 在 PRO5000 + SGLang 5.14 环境适配与 README。
- #17（2026-09-04 merged）：DeepSeek-V4 H20 kernel benchmark 数据。
- #19（2026-09-04 merged）：`fix: honor measured MLA decode latency`。这是最近一个纯逻辑修复，也是唯一的测试来源：把 `MLA.decode_attn_core` 从“解析式 MFU”改为“有实测点时直接用实测延迟”，并补了 3 个测试。作者 Gabriel，带 `Co-authored-by: TRAE Agent`。说明 maintainer 接受外部小修复，且接受 AI 辅助的 commit 标注。
- 未 merge 关闭的 #9/#13/#15/#16：多为 DeepSeek-V4 适配的规划或中间 PR，未合入的原因未在标题中说明，读 PR 正文需逐个核对，不宜猜测。

## 9. Fork 落差：13 个 commit

Fork `main`（`cbbc22b`，2026-01-04）到上游 HEAD（`d02a40c`，2026-09-04）之间的上游 commit：

```
fbc5098 Update argument help messages
76e74c3 update readme
7ed3db7 Support qwen3.5
3e99354 Add mfu log
0d2b567 Update readme
58c8bc8 Adapt qwen3.5-35B-A3B-FP8 (#5)
86bc550 Add benchmark data and HF config for Qwen3.5-122B-A10B (#10)
f2293da Adapt Qwen3.5-122B-A10B: TP support and model updates (#11)
9fd022d Adapt Qwen3.5-35B-A3B-FP8 in PRO5000 SGLang5.14 env (#12)
96527b4 Update pro5000 readme (#14)
ae3f73d Update README
457045d fix: honor measured MLA decode latency (#19)
d02a40c Add DeepSeek-V4 H20 kernel benchmark data (#17)
```

实质内容：Qwen3.5 支持、TP 支持、PRO5000 与 H200/GB200 的扩展、CLI 参数（`--tp-size`、`--prefill-*-mfu`、`--decode-scheduler-overhead-ms`、`--enable-shared-expert-overlap`）、#19 的 MLA decode 实测延迟修复、DeepSeek-V4 bench 数据。Fork 上跑旧代码会缺 TP、缺 PRO5000、缺 #19 的测试，结论不可直接用于上游。

## 10. 贡献面评估（只做判断，不提交）

按“小而明确、已确认、能验证”的标准：

| 方向 | 判定 | 理由 |
|---|---|---|
| #6 的公式修正 | 最小且已确认，但落点需先问 | 代码已对，报告 PDF 不在仓库。需确认 maintainer 希望的载体（重发 PDF / 仓库内公式注释 / 回归测试）。直接提代码 PR 会修错地方 |
| #7 的 A100 bench data | 可做，但重 | 需有 A100 真机跑 `kernel_benchmark` 产出 CSV，再补 `hardware/gpu.py` 的 A100 条目。无 A100 则做不了 |
| #7 的单卡多模型 / KV 上限配置 | 中等，可做 | 把 `kvcache_mem = mem - params - 15 - 5` 的保留量或 KV 上限做成可配参数，解决“KV 占满全部剩余显存”的痛点。需设计参数语义并与 maintainer 对齐 |
| 测试扩充 | 可做，价值明确 | 全仓仅 3 个测试。`flops.py` 的公式、TP 切分、`get_expected_active_experts`、`tp_all_reduce` 都是纯函数，易写单测。#19 已立了“测试随修复”的先例 |
| CI | 可做，但需 maintainer 配合 | 无 Actions。加一个 `python -m unittest` 的 workflow 是常规贡献，但冷 repo 的 maintainer 是否接受需看反馈 |
| #3 的 DSA 接入主链路 | 不建议现在做 | 维护者明确说未准备好开源，且缺字段与 offload 模拟。属于大 feature，不是小修复 |
| 第 7 节的 10 个待核实点 | 逐个核实后可转 PR | 其中 #2、#3、#6、#7 是小而明确的候选，但每个都需先用实测或与 maintainer 确认是 bug 还是刻意假设，再动手 |

冷 repo 的现实：#6 已确认 6 个月无修复，#3 追问 6 个月无回复。提 PR 前应预期响应慢，优先选“不需要 maintainer 事先拍板、改动自带验证”的类型（测试、纯函数公式回归），而非需要设计对齐的大改动。

## 11. 阅读路径与复现

阅读顺序（约 2 小时）：

1. `README.md` + `main.py`：入口与参数。
2. `config/model_config.py` + `flops/flops.py`：形状与 FLOPs 公式是全仓的锚。
3. `models/model.py`：看 TTFT/TPOT 如何组装。
4. `layers/attn.py` + `layers/moe.py` + `mfu/mfu.py`：看延迟与 MFU 查表。
5. `comm/comm.py` + `params/params.py` + `kvcache/kvcache.py`：通信与显存账本。
6. `models/hybrid_model.py`：对照 `Model` 看加权差异。
7. `kernel_sim/sparse_mla_fp8.py` + 第 8 节 issue：看 DSA 的零散状态。

复现（仓库根目录运行，纯 CPU 即可，无需 GPU）：

```bash
python3 main.py --config-path hf_configs/qwen3-30B-A3B_config.json \
  --device-type H20 --world-size 4 --decode-bs 100 --decode-only

python3 main.py --config-path hf_configs/qwen3-30B-A3B_config.json \
  --device-type H20 --world-size 1 --max-prefill-tokens 16384 --prefill-only

python3 -m unittest tests.test_attn -v
```

注意：必须在仓库根目录运行，否则 `bench_data` 相对路径查不到，全部 fallback 到默认 MFU 且只打印 Warning，结果会 silently 变化。

## 12. 附录：关键公式对照

| 量 | 技术报告（第 3 页） | 代码 |
|---|---|---|
| `FLOPs_attn_core` | `4 n_h S_q S_kv d_h R_mask` | `flops.py`：`2·bs·D·S + 2·bs·S·D`；prefill 再 `/1.8`（`R_mask≈0.556`） |
| `FLOPs_qkvo_proj` | `4 (n_h + n_kv) d_h d_hidden`（漏 `S_q`，见 #6） | `flops.py` / `layers/attn.py`：均以 `bs`（即 `S_q`）为 GEMM 的 `m`，正确 |
| `t_attn_core` | `FLOPs_attn_core / (FLOPS_gpu × MFU_attn_core)` | 同左；decode 有实测点时直接用实测延迟（#19） |
| `t_attn_proj` | `FLOPs_qkvo_proj / (FLOPS_gpu × MFU_attn_proj)` | `get_gemm_mfu_and_latency`：查 MFU 后套用 |
| `t_attn` | `t_attn_core + t_attn_proj` | `Model` / `HybridModel`：各段串行相加（TBO 除外） |
