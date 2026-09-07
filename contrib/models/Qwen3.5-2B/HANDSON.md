# Qwen3.5-2B on Trainium 上手实操

<!-- meta: description: Step-by-step Chinese hands-on for running Qwen3.5-2B
(hybrid gated-DeltaNet + GQA, vision-language) on Trainium2 with NxD Inference:
environment, checkpoint, text inference, TP selection, Neuron vision encoder,
accuracy against HuggingFace, and the errors you will hit. -->

<!-- meta: keywords: Neuron, Trainium, trn2, NxD Inference, Qwen3.5-2B, DeltaNet,
linear attention, GQA, mRoPE, vision language, VL, NKI, tensor parallel, 上手, 教程 -->

<!-- meta: date_updated: 2026-09-07 -->

<!-- Content type: procedural-tutorial -->

这篇是**动手复现**用的:从一台干净的 trn2 实例开始,把 `Qwen/Qwen3.5-2B` 的**纯文本**和
**图文**两条路都跑通,并且理解每个数字是怎么来的。模型本身的结构、完整测量矩阵和限制见
[README.md](README.md),这里只讲怎么做、会踩什么坑。

## 0. 本文的验证环境(和 README 的差别)

README 的数字是在 **trn2.48xlarge、TP=8** 上测的。本文**全部命令实际跑在
trn2.3xlarge(4 个逻辑核)上**,所以**统一用 TP=4**——这也是这台机器的上限,原因见第 5 节。
两边能对上的地方我都标了出来。

| | 本文 | README |
|---|---|---|
| 实例 | trn2.3xlarge(12 vCPU / 124 GB 内存 / 4 个逻辑核) | trn2.48xlarge |
| TP | **4** | 8(另有 TP=4 一组) |
| 日期 | 2026-09-07 | — |

工具链版本(和 README 的兼容性表一致,这点很重要,见 9.1):

```
neuronx-cc 2.26.6360.0 | nki 0.5.0 | neuronx-distributed 0.19.28492
torch-neuronx 2.9.0.2.15 (torch 2.9.1) | libneuronxla 2.2.17544
transformers 4.57.6 | Python 3.12
```

---

## 1. 先建立四个概念

**① 这个模型的 24 层不是同一种层。** 结构是 **[3 个门控 DeltaNet(线性注意力) + 1 个
完整 GQA 注意力] × 6**。DeltaNet 层是循环式的(conv1d + delta rule),状态大小与序列长度
**无关**;只有 6/24 层是真正的 KV-cache 注意力。这直接解释了后面两个观测:TTFT 对 prompt
长度不敏感,而 TPOT 由 all-reduce 和 host 循环开销主导。

**② DeltaNet 的实现是 NKI kernel,而且有多个版本。** `src/nki_kernels/` 下有四个:
默认 CTE 路径走 `nki_deltanet_fused.py`,而**图文路径必须换成 `legacy_direct`**——fused
kernel 在真实视觉 embedding 上数值不稳定,会退化成重复 token。`run_vl_smoke.py` 已经
在 import 之前就把环境变量设好了,别去覆盖它(第 6 节)。

**③ 不需要改动 NxDI 库本身,但必须让仓库代码生效。** 这个 contrib 模型是自带
modeling 代码的(`contrib/models/Qwen3.5-2B/src/`),脚本靠 `sys.path` 把
`contrib/models/Qwen3.5-2B` 加进来后 `from src.modeling_qwen35 import ...`。所以
**必须从仓库根目录跑脚本**,或者自己把这个目录加进 `PYTHONPATH`(见 9.1)。

**④ TTFT 由 `--seq-len` 决定,不由 prompt 长度决定。** 脚本里
`enable_bucketing=False`,每个 prompt 都被 pad 到编译时那一个 `seq_len` 桶。第 4 节会
用两组实测把这件事说清楚。

---

## 2. 环境准备

### 2.1 激活 venv

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

### 2.2 克隆仓库,并让**这个仓库的**代码生效

```bash
cd ~
git clone https://github.com/qingzwang/neuronx-distributed-inference.git
cd neuronx-distributed-inference
git checkout qwen3.5-2b-hybrid-deltanet

# 如果 venv 里没有 NxDI,或者你想确保用的是这个仓库的版本
pip install -e . --no-deps
```

`--no-deps` 不能省:否则 pip 会按 `setup.py` 把 torch / torch-neuronx / neuronx-cc
重装成别的版本,把预装环境弄坏。

确认一下版本和路径:

```bash
python - <<'EOF'
import torch, torch_neuronx, neuronx_distributed as nxd
import neuronx_distributed_inference as nxdi
import transformers, nki
print("nxdi  :", nxdi.__file__)
print("torch :", torch.__version__, "| torch_neuronx:", torch_neuronx.__version__)
print("nxd   :", nxd.__version__, "| transformers:", transformers.__version__)
EOF
```

`neuronx-cc` 要落在 **2.26.6360.0**:

```bash
pip list | grep -E 'neuronx-cc|nki |neuronx-distributed'
```

### 2.3 确认硬件

```bash
/opt/aws/neuron/bin/neuron-ls
```

看两处:表头的 `logical-neuroncore-config: 2`(默认,本文用它),和 `NEURON CORES` 一列
——那是可用逻辑核数,也是 TP 的上限。本文这台机器是 4。

---

## 3. 下载 checkpoint

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3.5-2B', local_dir='/mnt/nvme/models/Qwen3.5-2B')"
```

实测 13 个文件、**4.3 GB**,单个 safetensors 分片。确认一下拿到的是对的架构:

```bash
python -c "import json; c=json.load(open('/mnt/nvme/models/Qwen3.5-2B/config.json')); \
  print(c['model_type'], c['architectures'])"
# qwen3_5 ['Qwen3_5ForConditionalGeneration']
```

`model_type=qwen3_5` 是 transformers 5.x 才认识的架构。**Neuron 这条路不需要
transformers 认识它**——modeling 代码在本仓库里,配置是直接读 `config.json` 的。只有
第 8 节拿 HF 当参照时才需要新版 transformers。

---

## 4. 纯文本:第一次跑

```bash
python contrib/models/Qwen3.5-2B/test/integration/run_text_smoke.py \
    --model-path    /mnt/nvme/models/Qwen3.5-2B \
    --compiled-path /tmp/qwen35_2b_tp4 \
    --tp 4 --seq-len 512 --max-new-tokens 32 \
    --prompt "The capital of France is"
```

实测输出(trn2.3xlarge,TP=4,bf16,seq_len=512):

```
[compile] → /tmp/qwen35_2b_tp4
[compile] done in 100.1 s
[load] ← /tmp/qwen35_2b_tp4
========================================================================
prompt : 'The capital of France is'
output : 'The capital of France is Paris.\nA. True\nB. False\n\n<think>\nThinking Process:\n\n1.  **Analyze the Request:** The user is presenting a statement'
n_new  : 32
TTFT   : 33.1 ms
TPOT   : 4.6 ms  (217.50 tok/s)
========================================================================
```

和 README 里 TP=8 的期望输出**是同一句话开头**(`Paris.\nA. True\nB. False`),说明模型
逻辑跑对了。第二次跑同一条命令会跳过编译(检测 `model.pt` 已存在),几十秒就能出结果。

### 4.1 TTFT 到底由什么决定

同一个编译产物,只改 prompt 长度:

```bash
python contrib/models/Qwen3.5-2B/test/integration/run_benchmark.py \
    --model-path /mnt/nvme/models/Qwen3.5-2B --compiled-path /tmp/qwen35_2b_tp4 \
    --prompt-lens 16 64 256 --max-new-tokens 64 --repeats 5
```

| prompt tokens | TTFT (ms) | TPOT (ms) | tok/s |
|---|---|---|---|
| 16 | 24.0 | 4.70 | 211.9 |
| 64 | 24.0 | 4.66 | 214.8 |
| 256 | 24.1 | 4.65 | 215.1 |

**三个数字一模一样**。因为 `enable_bucketing=False`,16 个 token 的 prompt 也被 pad 到
512。要验证这个解释,把 `--seq-len` 换成 1024 重新编译一份再测:

```bash
python contrib/models/Qwen3.5-2B/test/integration/run_text_smoke.py \
    --model-path /mnt/nvme/models/Qwen3.5-2B --compiled-path /tmp/qwen35_2b_tp4_s1024 \
    --tp 4 --seq-len 1024 --max-new-tokens 8
python contrib/models/Qwen3.5-2B/test/integration/run_benchmark.py \
    --model-path /mnt/nvme/models/Qwen3.5-2B --compiled-path /tmp/qwen35_2b_tp4_s1024 \
    --prompt-lens 16 256 512 --max-new-tokens 64 --repeats 5
```

| prompt tokens | TTFT (ms) | TPOT (ms) | tok/s |
|---|---|---|---|
| 16 | 42.9 | 4.78 | 209.3 |
| 256 | 42.8 | 4.73 | 211.5 |
| 512 | 42.9 | 4.70 | 212.6 |

依然完全持平,但整体从 24 ms 抬到 43 ms——**桶宽翻倍,TTFT 也差不多翻倍**。这一组和
README 的 TP=4 表(42.2 / 42.2 / 42.5 ms,TPOT 4.71–4.77,209.7–211.8 tok/s)几乎逐项
对上,而那是在 48xlarge 上测的:TP=4 就用 4 个核,和实例多大无关。

**实践含义**:如果你的 prompt 普遍很短,就别编译一个大 `seq_len`,那是纯浪费;要么把
`seq_len` 压到实际需要,要么打开 NxDI 的 bucketing(`enable_bucketing=True`,像第 6 节
VL benchmark 那样给一组桶)。

### 4.2 TPOT 为什么是 4.7 ms 而不是更低

2B 模型、batch=1,每步的算力需求很小,时间花在 **TP all-reduce + host 侧生成循环**上。
所以:TP 从 4 提到 8 只把 TPOT 从 4.7 降到 4.0(README 数据),不到 1.2×,远不是 2×。
DeltaNet 层还进一步削弱了 TP 的收益——它的状态是每头独立的小矩阵,切开之后每个核的活
更碎。

---

## 5. TP 怎么选(以及为什么这台机器只能 TP=4)

**TP 的上限是可用逻辑核数。** `neuron-ls` 的 `NEURON CORES` 就是它:trn2.3xlarge 在
默认 `logical-neuroncore-config: 2` 下是 4,trn2.48xlarge 是 32。

在 4 核机器上直接要 TP=8 会怎样:编译能过(编译是 host 侧的事),**加载时进程直接
abort**:

```
INFO:Neuron:Loading presharded checkpoints for ranks: 0...7
terminate called after throwing an instance of 'c10::Error'
terminate called recursively
Aborted (core dumped)          # exit code 134
```

没有一句人话解释"核不够",所以**在跑之前先数核**。

### 5.1 想用 LNC=1 凑出 8 个逻辑核?不行(实测)

一个自然的想法:trn2.3xlarge 有 4 个物理核,`logical-neuroncore-config: 1` 下能看到
8 个逻辑核,那 TP=8 是不是就能跑了?我把 `run_text_smoke.py` 里的
`logical_nc_config=2` 改成 1,并设 `NEURON_LOGICAL_NC_CONFIG=1`,结果编译能过,加载报:

```
ERROR NRT:nrt_load_util  Mismatch detected between Runtime configuration and NEFF.
  Runtime currently configured with `NEURON_LOGICAL_NC_CONFIG=1` but NEFF
  /tmp/nxd_model_lnc1/context_encoding_model/_tp0_bk0/model.MODULE_....neff
  was compiled with `--lnc=2`. To resolve, either recompile the NEFF with `--lnc=1`
  or re-run with `NEURON_LOGICAL_NC_CONFIG=2`.
RuntimeError: Could not load the model status=2 message=Invalid
```

原因在
[`modeling_qwen35.py::get_compiler_args`](src/modeling_qwen35.py)(约 8136 行):这个模型
**自己覆写了编译参数,而且没有传 `--lnc`**,于是 NEFF 拿的是 builder 的默认值(trn2 上
是 2),`logical_nc_config=1` 只改了 NxDI 侧的记账,没改真正下发的编译 flag。

**结论:这台 4 核机器上就用 TP=4;要 TP=8 得有 8 个 LNC=2 的逻辑核(即 48xlarge 那类实例)。**

（顺带,第一次试 LNC=1 时我还撞上另一件事:NEFF 是从**默认编译工作目录
`/tmp/nxd_model`** 里复用的,里面还是上一次 `--lnc=2` 的产物,报的也是同一个 mismatch。
同时跑多个任务、或者换编译配置时,给每次跑设一个独立的 `BASE_COMPILE_WORK_DIR`。）

---

## 6. 图文(VL):图片 → 文字

### 6.1 先跑通(CPU 视觉编码器)

```bash
python contrib/models/Qwen3.5-2B/test/integration/run_vl_smoke.py \
    --model-path    /mnt/nvme/models/Qwen3.5-2B \
    --compiled-path /tmp/qwen35_2b_vl_tp4 \
    --image         examples/cat.png \
    --prompt        "What is in this image? Describe it briefly." \
    --tp 4 --seq-len 2048 --max-new-tokens 48
```

实测(1024×1024 的猫图):

```
[compile-text] done in 137.4 s
[load-vision] loading CPU vision encoder weights
[input] input_ids shape=torch.Size([1, 1048])
[input] pixel_values shape=torch.Size([4096, 1536])
[input] image_grid_thw=[[1, 64, 64]]
prompt      : 'What is in this image? Describe it briefly.'
n_new       : 48
elapsed     : 2.8 s
generated   : 'This is a casual, home-style photograph featuring a **black cat**
               lounging comfortably on a patterned cushion or pillow. ...'
```

几件事值得对着数字看:

- **1024×1024 → 4096 个 patch → 1024 个视觉 token**(`patch_size=16` 给 64×64=4096 个
  patch,`spatial_merge_size=2` 把 2×2 合成 1,得 1024)。`input_ids` 长 1048 = 1024 个
  图像占位 token + 24 个文本 token。
- 所以 **`--seq-len` 必须放得下"视觉 token + 文本 token"**。1024px 的图至少要 2048。
  2048px 的图会产生 4096 个视觉 token,得上 8192 的桶。
- VL 的文本模型要用 `use_text_only_cte_inputs=False` 重新编译(它多了
  vision_embeddings / vision_mask / mRoPE 三组输入),所以**不能复用第 4 节的产物**。

`run_vl_smoke.py` 在 import 之前设了两个环境变量:

```python
os.environ.setdefault("QWEN36_DELTANET_CTE_IMPL", "legacy_direct")
os.environ.setdefault("QWEN36_DELTANET_MULTIHEAD_CTE", "0")
```

**不要覆盖它们。** 默认的 fused-multihead DeltaNet kernel 在真实视觉 embedding 上会
退化(输出重复 token),legacy-direct 是稳定的那条。

### 6.2 把视觉编码器也搬到 Neuron 上

CPU 那条路在小实例上尤其吃亏:trn2.3xlarge 只有 **12 个 vCPU**,ViT 全靠它算。先编译
Neuron 版本(按 patch-token 数分桶):

```bash
python contrib/models/Qwen3.5-2B/test/integration/compile_vision_encoder.py \
    --model-path /mnt/nvme/models/Qwen3.5-2B \
    --out-dir    /tmp/qwen35_2b_vl_bench/vision \
    --buckets    1024 4096
```

实测编译耗时:**bucket 1024 用 132 s,bucket 4096 用 566 s**,产物各约 1 GB
(`vision_encoder_1024.pt` / `vision_encoder_4096.pt`)。

benchmark 脚本读 `/tmp/test_image_<size>.jpg`,自己先准备:

```bash
python - <<'EOF'
from PIL import Image
src = Image.open('examples/cat.png').convert('RGB')
for s in (512, 1024, 2048):
    src.resize((s, s), Image.LANCZOS).save(f'/tmp/test_image_{s}.jpg', quality=95)
EOF
```

CPU 基线(注意这次打开了 bucketing,`--buckets 512 1024 2048`):

```bash
python contrib/models/Qwen3.5-2B/test/integration/run_vl_benchmark.py \
    --model-path /mnt/nvme/models/Qwen3.5-2B --compiled-path /tmp/qwen35_2b_vl_bench \
    --tp 4 --images 512 1024 --buckets 512 1024 2048 \
    --max-new-tokens 48 --repeats 3
```

再加 `--vision-compiled-dir /tmp/qwen35_2b_vl_bench/vision --skip-compile` 跑 Neuron 版本。
实测(TP=4,48 个新 token,3 次中位数):

| 图片 | 视觉 token | 视觉编码器 | TTFT | TPOT | 提速 |
|---|---|---|---|---|---|
| 512×512 | 256 | CPU | 405.0 ms | 3.72 ms | — |
| 512×512 | 256 | **Neuron(TP=1)** | **145.4 ms** | 4.78 ms | **2.8×** |
| 1024×1024 | 1024 | CPU | 2361.8 ms | 4.33 ms | — |
| 1024×1024 | 1024 | **Neuron(TP=1)** | **777.9 ms** | 4.83 ms | **3.0×** |

README 在 48xlarge 上量到的是 2.0–2.1×;这里更高(2.8–3.0×),**因为分子变大了**:
48xlarge 有 192 个 vCPU,3xlarge 只有 12 个,CPU 那条路慢得多。**实例越小,把 ViT 搬上
Neuron 的收益越大。**

TPOT 在两条路上一样(4.8 vs 3.7–4.3,差异在抖动量级),因为文本解码本来就全在 Neuron 上,
和视觉编码器在哪没关系。

**输出是否一致?** 两条路的 48 个 token **逐字符完全相同**(512 和 1024 都是),这是
traced 编码器和 CPU 编码器数值一致的最直接证据:

```python
json.load(open('cpu.json'))['results'][i]['text_sample'] == \
json.load(open('neuron.json'))['results'][i]['text_sample']   # True
```

### 6.3 再把视觉编码器切到多个核上

```bash
python contrib/models/Qwen3.5-2B/test/integration/compile_vision_encoder_tp.py \
    --model-path /mnt/nvme/models/Qwen3.5-2B \
    --out-dir    /tmp/qwen35_2b_vl_bench/vision \
    --tp 4 --buckets 4096
```

编译 **299.9 s**,产物在 `vision/bucket_4096/tp_{0..3}.pt`。同一个目录下 TP 产物**优先于**
旧的 `vision_encoder_4096.pt`,所以 benchmark 命令不用改,重跑即可:

| 1024×1024 的视觉路径 | E2E TTFT | 相对 |
|---|---|---|
| CPU | 2361.8 ms | 1.00× |
| Neuron,TP=1 | 777.9 ms | 3.0× |
| **Neuron,TP=4** | **573.9 ms** | **4.1×**(相对 TP=1 再快 1.36×) |

README 在 48xlarge 上是 526 → 318 ms(1.65×);这里 778 → 574 ms(1.36×)。差别在于本文
文本模型也占着同样这 4 个核,分片视觉编码器和它抢资源。

### 6.4 精度上的现实

同一张猫图(黑白花色的猫躺在沙发上,旁边有花纹靠垫和一个遥控器):

- **1024×1024**:CPU 和 Neuron 都说 "**black cat** lying on a patterned pillow" ——对。
- **512×512**:CPU 和 Neuron **都**说 "a **person lying on their back** on a textured
  brown couch" ——错,而且两条路错得一模一样。

也就是说,512px 认不出来是**模型在低分辨率下的能力问题**,不是 Neuron 移植的问题——这一点
只有把两条路的输出摆在一起才能下结论,所以 6.2 里那个"逐字符一致"很重要。想让 VL 更准,
先加分辨率(代价是视觉 token 数平方增长,`seq_len` 桶和 TTFT 跟着涨),而不是去调 kernel。

---

## 7. 跑测试

```bash
QWEN35_MODEL_PATH=/mnt/nvme/models/Qwen3.5-2B \
QWEN35_COMPILED_PATH=/tmp/qwen35_2b_tp4 \
QWEN35_TP_DEGREE=4 QWEN35_SEQ_LEN=512 \
python -m pytest contrib/models/Qwen3.5-2B/test/integration/test_model.py -q
```

```
1 passed, 1 skipped, 13 warnings in 39.84s
```

`test_generation_and_latency` 过;`test_accuracy_vs_hf` 被跳过,因为它要 HF 参照——下一节。
注意它默认 `QWEN35_TP_DEGREE=8`,在 4 核机器上必须显式传 4,否则就是第 5 节那个 core dump。

---

## 8. 和 HuggingFace 对精度

`transformers 4.57.6` 不认识 `qwen3_5`,所以 HF 参照要**另开一个 venv**:

```bash
python3 -m venv /tmp/hf_ref_venv
/tmp/hf_ref_venv/bin/pip install "transformers==5.13.0" "torch>=2.6" \
    safetensors sentencepiece accelerate
```

（实测装出来是 transformers 5.13.0 + torch 2.14.0。这个 venv **不要**用来跑 Neuron。)

两侧各自 dump:

```bash
# Neuron 侧(在 NxDI venv 里)
python contrib/models/Qwen3.5-2B/test/integration/run_accuracy_check.py \
    --model-path /mnt/nvme/models/Qwen3.5-2B --compiled-path /tmp/qwen35_2b_tp4 \
    --max-new-tokens 16 --skip-hf --out-json /tmp/qwen_neuron_acc.json

# HF 侧(CPU,bf16 greedy)
/tmp/hf_ref_venv/bin/python contrib/models/Qwen3.5-2B/test/integration/run_hf_reference.py \
    --model-path /mnt/nvme/models/Qwen3.5-2B --max-new-tokens 16 \
    --out-json /tmp/qwen_hf_ref.json
```

实测逐 token 对比(TP=4,16 个新 token):

| prompt | 一致 token |
|---|---|
| "The capital of France is" | 16 / 16 |
| "The largest planet in our solar system is" | 16 / 16 |
| "Water boils at" | 3 / 16 |
| "A haiku about autumn leaves:" | 15 / 16 |
| "In one sentence, explain photosynthesis." | 16 / 16 |

合计 **66/80 = 82%**,5 个 prompt 里 **3 个完全一致**。两处分歧长这样:

```
Water boils at
  neuron: 'Water boils at $100^{\circ} \mathrm{C}$ at sea level'
  hf    : 'Water boils at 100°C at sea level. If the pressure is 10'

A haiku about autumn leaves:
  neuron: '...The wind blows soft,\nThe leaves turn gold and'
  hf    : '...The wind blows soft,\nThe leaves turn gold,'
```

**怎么读这个结果**:第一条是**一个 token 分叉之后的级联**——模型在"用 LaTeX 写 100°C"和
"直接写 100°C"之间本来就势均力敌,一步走岔后面全不同,但两句话都是对的。第二条只在第 16
个 token 上分叉(`and` vs `,`),再多给几个 token 大概率还是同义的句子。这是 **bf16 累加
路径对 bf16 CPU 实现**的正常量级,不是移植缺陷;要更严格的判据,应该对 logits 而不是
贪心 token 序列。

（README 在 TP=8 上记的是 53/80 = 66%、3/5 完全一致。同一个模型不同 TP、不同的 sum 顺序,
落在同一个量级——这也说明这个指标本身就有几个百分点的抖动,别把它当回归门槛。）

---

## 9. 常见问题

### 9.1 `ModuleNotFoundError: No module named 'src'`

脚本用相对自身位置的 `sys.path.insert` 找 `src`,所以**把脚本复制到别处跑就会断**。
要么从仓库根目录跑原路径,要么:

```bash
export PYTHONPATH=$HOME/neuronx-distributed-inference/contrib/models/Qwen3.5-2B:$PYTHONPATH
```

### 9.2 其他

| 现象 | 原因 | 处理 |
|---|---|---|
| `terminate called after throwing an instance of 'c10::Error'` + core dump,发生在 `Loading presharded checkpoints for ranks: 0...N` 之后 | 要的 TP 大于可用逻辑核数 | `neuron-ls` 数 `NEURON CORES`,把 `--tp` 降下来(第 5 节) |
| `Mismatch detected between Runtime configuration and NEFF ... compiled with --lnc=2` | 运行时 LNC 和 NEFF 编译时的 `--lnc` 不一致;也可能是复用了 `/tmp/nxd_model` 里旧配置的 NEFF | 不要动 LNC(第 5.1 节);换配置时给独立的 `BASE_COMPILE_WORK_DIR` |
| `FileNotFoundError: /tmp/test_image_512.jpg` | `run_vl_benchmark.py` 不接受 `--image`,它按尺寸找固定路径 | 自己生成 `/tmp/test_image_<size>.jpg`(第 6.2 节) |
| VL 输出是重复的 token | fused DeltaNet kernel 在视觉 embedding 上不稳定 | 保持 `QWEN36_DELTANET_CTE_IMPL=legacy_direct`,别覆盖 |
| VL 报输入超出 CTE 桶 | 视觉 token 把总长顶过了 `seq_len` | 1024px 至少 2048;2048px 要 8192 |
| `test_accuracy_vs_hf` 被 skip | 没有 transformers ≥ 5.13 的参照 venv | 第 8 节 |
| pytest 在 4 核机器上 core dump | 它默认 `QWEN35_TP_DEGREE=8` | 显式传 `QWEN35_TP_DEGREE=4` |
| `blockwise_mm_bwd` 导入告警 | MoE 专用 NKI kernel,这个模型用不到 | 无害 |
| 编译反复从头开始 | `--compiled-path` 换了,或 `model.pt` 被删 | 固定 `--compiled-path`;改 TP / seq_len / VL 开关都必然重新编译 |

### 9.3 一次跑完整套的顺序

```
第 4 节 文本(TP=4, seq 512)         编译 100 s  →  TTFT 33 ms / TPOT 4.6 ms
第 4.1 节 seq 1024 对照              编译 ~100 s →  TTFT 43 ms
第 6.1 节 VL smoke(seq 2048)        编译 137 s  →  48 token / 2.8 s
第 6.2 节 视觉编码器编译             132 s + 566 s
第 6.2 节 VL benchmark ×2            文本 153 s + 两轮测量
第 6.3 节 视觉编码器 TP=4            300 s
第 8 节 HF 参照 venv + 两侧 dump     装包几分钟 + CPU 生成几分钟
```

---

## 10. 接下来

- [README.md](README.md) —— 模型结构、完整测量矩阵(含 TP=8 和 2048px 分块)、兼容性表
- `src/modeling_qwen35.py` —— DeltaNet + GQA 解码器,`get_compiler_args` 在这里(第 5.1 节)
- `src/nki_kernels/` —— 四个 DeltaNet kernel,以及 QK-norm+RoPE kernel
- `src/modeling_qwen35_vision.py` —— ViT 包装:`load_cpu_model` / `load_compiled` /
  `_tiled_forward`(2048px 的 2×2 分块走这里)
