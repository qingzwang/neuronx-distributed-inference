# 在 Trainium 上跑 FLUX 动态 LoRA —— 上手指南

这篇是给没用过 Trainium 的人写的：从一台干净的机器开始，一步步做到「编译一次
FLUX，之后随时换不同的 LoRA 出图」。全部命令都在 **trn2.3xlarge** 上实际跑过
（2026-09-03），输出也是真实输出。

英文版的特性说明和完整数据在 [flux_lora.md](flux_lora.md)，这里只讲怎么动手。

---

## 0. 先建立三个概念

**① Neuron 是「先编译、再执行」的。** 模型不是直接跑 PyTorch，而是先编译成 NEFF
（设备可执行文件），再加载到 Neuron 核上。所以改分辨率、改并行度、改 LoRA 槽位数
量，都要重新编译。第一次编译 1024×1024 的完整 FLUX 流程要 **15–30 分钟**（最慢的
是 VAE decoder），之后复用同一个 `compile_workdir` 就只需要加载，约 3–4 分钟。

**② 一个进程占一个逻辑核。** trn2.3xlarge 是 1 颗 Trainium2，默认 LNC=2，也就是
**4 个逻辑核，每核约 22 GiB HBM**。本文用 TP=4，刚好把这颗芯片用满。

**③ 换 LoRA 不需要重新编译。** 这就是「动态 LoRA」的含义：编译时在图里留下
`adapter_ids` 这个输入和固定数量的 LoRA 槽位，运行时只是把权重搬进槽位、然后告诉
图用哪个槽。加载一个编译时完全没提过的 LoRA 也可以。

---

## 1. 确认机器和驱动

```bash
export PATH=/opt/aws/neuron/bin:$PATH
neuron-ls
```

能列出设备就说明驱动和 runtime 正常。如果 `neuron-ls` 找不到，说明这台机器不是
Neuron 实例，或者驱动没装（用 Deep Learning AMI 一般都带）。

---

## 2. 建一个专用 venv

NxD Inference 用的是 **torch-neuronx + neuronx-distributed** 这一套。注意它和 vLLM
Neuron 插件用的 `libtorch-neuronx-lite` 那一套 **互相冲突**（torch、torch-xla、
transformers 版本都不一样），所以别混在一个环境里，各建一个 venv。

```bash
python3 -m venv /mnt/nvme/venv-nxdi
/mnt/nvme/venv-nxdi/bin/pip install --upgrade pip

# 主体
/mnt/nvme/venv-nxdi/bin/pip install "neuronx-distributed-inference==0.9.*" \
    --extra-index-url https://pip.repos.neuron.amazonaws.com

# 必须：pip 会把 neuronx-cc 解析到 2.27，这个版本编译 FLUX 会崩（见第 9 节）
/mnt/nvme/venv-nxdi/bin/pip install "neuronx-cc==2.26.6360.0" \
    --extra-index-url https://pip.repos.neuron.amazonaws.com

# 用本仓库的代码替换掉 pip 装的那份
/mnt/nvme/venv-nxdi/bin/pip install -e /path/to/neuronx-distributed-inference --no-deps

# FLUX 需要
/mnt/nvme/venv-nxdi/bin/pip install "diffusers==0.32.0" accelerate

# 只有你想跟 CPU 对照验精度时才需要（第 8 节）
/mnt/nvme/venv-nxdi/bin/pip install peft
```

验证过的版本组合：

```
torch 2.9.1 | torch-xla 2.9.0 | torch-neuronx 2.9.0.2.15.32035
neuronx-distributed 0.19.28492 | libneuronxla 2.2.17544 | neuronx-cc 2.26.6360.0
transformers 4.57.6 | diffusers 0.32.0 | Python 3.12.3
```

之后所有命令都在这个环境里跑：

```bash
export PATH=/mnt/nvme/venv-nxdi/bin:/opt/aws/neuron/bin:$PATH
```

---

## 3. 下载模型和两个 LoRA

FLUX.1-dev 是 gated 仓库，要先在网页上同意协议，然后登录：

```bash
hf auth login          # 老版本是 huggingface-cli login

# 底模，约 24 GB（含两个 text encoder 和 VAE）
hf download black-forest-labs/FLUX.1-dev --local-dir /models/FLUX.1-dev

# LoRA 一：XLabs 格式，rank 16，22 MiB，只改双流块的注意力投影
hf download XLabs-AI/flux-RealismLora lora.safetensors \
    --local-dir /adapters/xlabs-realism

# LoRA 二：kohya 格式，rank 64，585 MiB，几乎所有线性层都改
hf download strangerzonehf/Flux-Super-Realism-LoRA super-realism.safetensors \
    --local-dir /adapters/super-realism
```

特意选了两个**格式和 rank 都不同**的 LoRA。社区 FLUX LoRA 至少有三种命名约定
（diffusers/PEFT、kohya、XLabs），三种都支持——文件会先交给
`FluxPipeline.lora_state_dict`，让 diffusers 自己的转换器处理格式，然后再映射到
NxDI 的模块名上。你不需要关心自己下的 LoRA 是哪种。

---

## 4. 第一次跑

```bash
cd /path/to/neuronx-distributed-inference

python examples/generate_flux_lora.py \
    -c /models/FLUX.1-dev \
    --compile_workdir /tmp/flux-lora/ \
    --lora xlabs=/adapters/xlabs-realism \
    --dynamic-lora kohya=/adapters/super-realism/super-realism.safetensors \
    --max-lora-rank 64 \
    --max-loras 1 \
    -hh 1024 -w 1024 -n 20 \
    --save_image
```

- `--lora` = **编译时就声明**的 LoRA，加载完就在设备槽位里。
- `--dynamic-lora` = **加载之后**再塞进来的 LoRA，用来验证「编译时没提过也能用」；
  给了这个参数就会自动打开 `dynamic_multi_lora`。
- `--max-loras 1` = 设备上同时只放 1 个 LoRA（外加一个 base 槽）。故意设小，这样
  两个 LoRA 会互相挤，你能直接看到换入的代价。真实部署里应该设成能装下热点 LoRA
  的数量。

第一次会编译（15–30 分钟），之后再跑同一个 `--compile_workdir` 就会打印
`already compiled at ..., skipping compilation` 直接加载（3–4 分钟）。

输出（这是 256px / 4 步跑出来的真实输出，路径按本文的约定改写过；1024px 只是数字更大）：

```
loaded kohya from /adapters/super-realism/super-realism.safetensors in 1.14 s
adapters available: ['kohya', 'xlabs']
base                       0.37 s
xlabs                      0.36 s
kohya                      2.18 s
```

以及三张图 `output_base.png` / `output_xlabs.png` / `output_kohya.png`。三个 LoRA 在
同一提示词、同一种子下的效果对比见 [flux_lora.md 的 Samples](flux_lora.md#samples)。

---

## 5. 这几个数字怎么读

| 行 | 含义 |
|---|---|
| `loaded ... in 1.14 s` | 磁盘 → 主机内存。每个 LoRA 只付一次，时间跟文件大小成正比（22 MiB 的那个只要 0.17 s） |
| `base 0.37 s` | 不用 LoRA。base 永远占 0 号槽，随时可用 |
| `xlabs 0.36 s` | 这个 LoRA 已经在设备槽位里，**和不用 LoRA 一样快**——槽位号是图的输入，选它不花钱 |
| `kohya 2.18 s` | 只有 1 个槽，kohya 得把 xlabs 顶掉，多出来的 ~1.8 s 就是主机 → 设备的搬运 |

1024px 下的完整数据（中位数，`max_loras=1` 所以每次换 LoRA 都必然 miss）：

| 步数 | 不用 LoRA | LoRA 已在槽位 | LoRA 需要换入 | 换入代价 | 占比 |
|---|---|---|---|---|---|
| 4 | 1.58 s | 1.57 s | 3.37 s | 1.80 s | 53% |
| 20 | 6.39 s | 6.38 s | 8.15 s | 1.77 s | 22% |
| 28 | 8.80 s | 8.81 s | 10.58 s | 1.77 s | 17% |

两个要点：

1. **命中是真免费**：第二列和第三列在任何步数下都测不出差别。
2. **换入是每请求一次，不是每步一次**：绝对值恒定在 ~1.78 s，所以步数越多它占的
   比例越小。想彻底消掉它，就把 `max_loras` 调大让热点 LoRA 常驻。

---

## 6. 在自己代码里怎么写

### 编译时就声明（最简单）

```python
import torch
from neuronx_distributed_inference.models.diffusers.flux.application import (
    NeuronFluxApplication, create_flux_config, get_flux_parallelism_config,
)
from neuronx_distributed_inference.models.diffusers.flux.lora import build_flux_lora_config

lora_config = build_flux_lora_config(
    max_loras=2,        # 设备上常驻几个（base 槽另算）
    max_lora_rank=64,   # 槽位按这个宽度开，见第 7 节
    lora_ckpt_paths={"realism": "/adapters/xlabs-realism"},
)

tp = 4
clip_c, t5_c, backbone_c, vae_c = create_flux_config(
    "/models/FLUX.1-dev", get_flux_parallelism_config(tp), tp,
    torch.bfloat16, 1024, 1024, lora_config=lora_config,
)
app = NeuronFluxApplication(
    model_path="/models/FLUX.1-dev",
    text_encoder_config=clip_c, text_encoder2_config=t5_c,
    backbone_config=backbone_c, decoder_config=vae_c,
    height=1024, width=1024,
)
app.compile("/tmp/flux-lora/")
app.load("/tmp/flux-lora/")

app.set_lora_adapters("realism")      # 之后的生成都用它
image = app(prompt="...", num_inference_steps=20).images[0]

app.set_lora_adapters(None)           # 回到底模
```

### 运行时再加

```python
lora_config = build_flux_lora_config(
    max_loras=1,
    max_cpu_loras=4,            # 主机内存里备着几个
    max_lora_rank=64,
    dynamic_multi_lora=True,    # 关键
    lora_ckpt_paths={"realism": "/adapters/xlabs-realism"},   # 可以完全不给
)
# ... 同上 compile / load ...

app.add_lora_adapter("superreal", "/adapters/super-realism/super-realism.safetensors")
app.set_lora_adapters("superreal")    # 第一次用到时才搬进设备槽位
print(app.list_lora_adapters())       # {'realism', 'superreal'}
```

### 为什么是 `set_lora_adapters` 而不是传参

`FluxPipeline.__call__` 没有 adapter 参数，也不会把多余的 kwargs 往 transformer
传，所以走完整文生图流程时没有别的办法指定 LoRA。如果你是**直接调 backbone**
（比如自己写去噪循环、或者做精度对照），那就可以每次传：

```python
out = app.pipe.transformer(
    hidden_states=..., timestep=..., guidance=...,
    pooled_projections=..., encoder_hidden_states=...,
    txt_ids=..., img_ids=..., return_dict=False,
    adapter_ids=["superreal"],        # 每个 batch item 一个名字；一个名字会广播
)
```

---

## 7. 三级缓存和三个参数

```
设备（HBM）   max_loras 个槽位，adapter_ids 直接选，不搬数据
    ↑ 换入 ~1.78 s（每请求一次）
主机（内存）  max_cpu_loras 个，已经切分好、随时可搬
    ↑ 加载 0.17–1.4 s（每个 LoRA 一次，跟文件大小成正比）
磁盘         add_lora_adapter() 读文件、切分、放进主机层
```

**`max_lora_rank` 是最重要的一个参数。** 槽位是按它开的，跟 LoRA 自己的 rank 无
关：rank 8 的 LoRA 塞进 rank 64 的槽位会被零填充，内存和换入时间都按 rank 64 算。
实测（同一个 rank-16 的 LoRA，只改编译时的槽宽）：

| `max_lora_rank` | 每槽每核显存 | 换入耗时 |
|---|---|---|
| 16 | 158.5 MB | 940 ms |
| 32 | 317 MB | 1322 ms |
| 64 | 634 MB | 1754 ms |

显存严格线性；换入时间不线性，因为里面有大约 0.7 s 是固定开销——一次换入要发起
约 4300 次独立拷贝（每个被适配的模块、每个 rank 一次），跟每次拷多大关系不大。所以
从 64 降到 16，显存省 4 倍，换入只快 1.9 倍。

结论：**按你实际要用的最大 rank 来设，别图省事写 128**——省的主要是显存，而显存又
直接决定你能常驻几个 LoRA，也就是能不能完全绕开换入。

**`max_loras`** 决定有多少 LoRA 能常驻。命中免费，所以热点 LoRA 全放进去是最划算
的优化；代价是每槽每核的显存。构建时 runtime 会把总量打印出来：

```
WARNING The memory footprint for LoRA adapters on each Neuron core is 1902.375 MB
```

（这里是 1 个 LoRA + base 槽 + 1 份暂存缓冲 = 3 份。）显存预算要和分片后的 backbone
一起算：trn2 每核约 22 GiB。

**`max_cpu_loras`** 决定主机层能备多少。主机层放不下时会淘汰，下次再要就得**重新
读磁盘**——这是最贵的路径，所以按工作集大小给足。淘汰策略默认 LRU，也可以
`eviction_policy="lfu"`。

---

## 8. 自己验一遍精度

思路：让 Neuron 和 CPU 上的 diffusers **加载同一个 LoRA、喂同样的输入**，比较一步
去噪的输出（velocity）。只比一步，因为多步会把误差混进采样器，看不出问题出在哪。

```python
from diffusers import FluxPipeline
ref = FluxPipeline.from_pretrained(CKPT, torch_dtype=torch.float32)   # 注意是 torch_dtype
ref.load_lora_weights("/adapters/super-realism/super-realism.safetensors")
# 用同一份 latents / embeds / timestep 分别过 ref.transformer 和 app.pipe.transformer，
# 再算 cosine similarity
```

两个坑（都踩过）：

- diffusers 0.32 接的参数叫 `torch_dtype`，写 `dtype=` 会被 `**kwargs` 悄悄吃掉，
  你会拿到一个 fp32 的参考却以为是 bf16。
- 固定种子的 `randn_tensor` 在 fp32 和 bf16 下抽出来的是**不同的数**，不是同一组数
  四舍五入。要么显式把同一份 latents 传给两边，要么别用种子对比。

我实测的结果（TP=4，对照 CPU fp32，每个槽位都跟三个参考比）：

| 槽位 | vs CPU 底模 | vs CPU xlabs | vs CPU kohya |
|---|---|---|---|
| 0（base） | **0.99923** | 0.98942 | 0.97726 |
| 1（xlabs） | 0.98918 | **0.99949** | 0.97403 |
| 2（kohya） | 0.97893 | 0.97529 | **0.99960** |

0.9992 就是 bf16 相对 fp32 的噪声底，每个槽位对自己的参考都贴在这个底上，对别人的
参考明显低——说明槽位没读错权重。另外 xlabs 被挤出设备槽位再换回来之后，输出**逐位
相同**（max|d| = 0）；1024px 20 步的完整出图也能跨进程复现出同样的 PNG 字节。

不需要设备的单元测试：

```bash
pytest test/unit/models/flux/test_flux_lora.py
FLUX_LORA_TEST_CHECKPOINT=/models/FLUX.1-dev pytest test/unit/models/flux/test_flux_lora.py
```

第二种写法会额外跑需要 checkpoint 的那几个（在 meta device 上建模型，不占显存也不
编译）。

---

## 9. 踩坑清单

**编译直接失败，报 `[NCC_ISMP902] Simplifier error: is_subset()`**
neuronx-cc 2.27 的问题，钉回 `2.26.6360.0`（第 2 节）。

**两个任务同时跑，其中一个编译到一半报 `neff_packager: failed to open output file`**
默认编译工作目录都是 `/tmp/nxd_model`，另一个进程（包括 `pytest
test/unit/models/flux/`）会把它清掉。同时跑多个任务时给每个设 `BASE_COMPILE_WORK_DIR`。

**`RuntimeError: The number of LoRA adapter IDs is N, but it should equal the prompts number`**
`adapter_ids` 的长度要等于 batch。传单个名字（字符串或长度 1 的列表）会自动广播，
超过 1 个就必须逐个对应。

**报 rank 超过 `max_lora_rank`**
槽位是编译时按 `max_lora_rank` 开的，装不下更宽的 LoRA。这是显式报错而不是静默截断
——按提示提高 `max_lora_rank` 重新编译。

**日志里刷出成千上万条 `scatter/gather ... out-of-bound access`**
你直接传了张量形式的 `adapter_ids` 且值越界。图本身不做边界检查，设备会越界读然后
返回垃圾。用名字而不是数字就不会碰到；确实要传数字的话，合法范围是
`0 .. max_loras-1`（注意 `enable_base_model_only` 会让 `max_loras` 自动 +1）。

**LoRA 明明设了却看起来没生效（走完整 pipeline 时）**
`FluxPipeline.__call__` 不转发额外 kwargs，必须用 `app.set_lora_adapters(...)`。

**警告说忽略了若干个非 transformer 的张量**
只有 backbone 支持 LoRA，text encoder 的 LoRA 会被丢掉。CLIP / T5 的图没有 LoRA 输入。

**`RuntimeError: Expected self.dtype() == dst.dtype() to be true`**
换入时主机缓冲和设备张量 dtype 不一致。原因是 FLUX 的线性层先按 fp32 建、之后整体
cast 成 bf16，LoRA 层却记下了 fp32。这个已经在
`lora.py::_align_lora_dtype` 修掉了；如果你还看到，说明代码不是最新的。

**HBM 不够（load 阶段 OOM）**
每核预算约 22 GiB，要装下分片后的 backbone 加上 `(max_loras + 1) × 每槽显存`。
先降 `max_lora_rank`，再降 `max_loras`。

**跑起来但只用到一个核 / collectives world size 不对**
一个进程一个逻辑核，用 `NEURON_RT_VISIBLE_CORES` 限定可见核数，让它和 TP 对齐，
例如 TP=2 时 `NEURON_RT_VISIBLE_CORES=0-1`。

---

## 10. 继续读

- [flux_lora.md](flux_lora.md) —— 特性说明、完整数据、限制
- [generate_flux_lora.py](generate_flux_lora.py) —— 本文用的脚本
- [generate_flux.py](generate_flux.py) —— 不带 LoRA 的 FLUX 示例
- [slora.md](slora.md) —— LLM 侧的多 LoRA 服务
