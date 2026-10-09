# V2 验证：工程测试与真实权重探针

日期：2026-10-09。V1 基线：`4b4201177cb8e7bb2facc44ef1891a7639ab7aa4`。

## 工程验证

环境：macOS ARM64，Python 3.12.13，PyTorch 2.6.0，transformers 4.57.6，datasets 3.6.0，pyarrow 23.0.0。使用 CPU；测试限制线程数以减少小张量的调度开销。

维护的测试集合共 **45 项**，已通过：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_vision_budget.py tests/test_dataset.py tests/test_metrics.py \
  tests/test_budget_training.py tests/test_budget_pipeline.py
```

覆盖 V1 行为回归、变长 padding、相邻图片压缩、有效答案对齐、KL 教师梯度隔离、短尾累积、整批/不等长微批梯度一致、CE/KD 两种完整训练入口、冻结参数、原生配置 sidecar 加载、paired bootstrap 与严格目标检查。两进程 CPU Gloo 测试验证 rank 有效 token 数不同时的 DDP 梯度与全局 token 均值一致；它使用小型语言模块，不能等同于 CUDA VLM 多卡训练认证。

完整训练入口测试使用真实 SigLIP 模块、真实 tokenizer 和临时 Parquet，但测试权重是随机小模型。小模型上 8 次优化后训练目标下降，只验证可训练性，不证明问答效果。

还执行了 Python compileall、`git diff --check`，新训练、配对比较、真实权重探针、QA 评测和原始推理共五个 CLI 的 `--help`。工作区中另外出现的带数字后缀副本没有纳入本次提交；对维护的五个测试文件单独核验，避免重复收集抬高测试数量。

## 原累积问题的最小数值复现

两个类别的共享 logits 为 `[0.3, -0.2]`；一个微批只有 1 个有效答案 token，目标类别为 0，另一个有 3 个，目标均为 1：

```text
旧的微批 loss 均值梯度： [ 0.12245935, -0.12245934]
全局有效 token 均值梯度：[ 0.37245935, -0.37245929]
最大绝对差：0.25
```

新入口先对整个窗口计数，再归一化 CE 与 KL；旧训练脚本保留历史逻辑，预算实验应切换到新入口。上述 0.25 是一个数值反例，不是训练 loss 的普遍改进幅度。

## 真实公开权重：已完成加载与生成

本次下载了公开资源，固定为以下 Hugging Face revision，没有调用付费教师 API：

| 资源 | revision | 使用文件 |
|---|---|---|
| jingyaogong/minimind-3v-pytorch | `83ae67a06ec29dc1acf2294d8edacbdef12f4013` | sft_vlm_768.pth |
| jingyaogong/siglip2-base-p32-256-ve | `9465d1dc89db6bc6227c5b6b0e0ca9b940325d62` | config.json、preprocessor_config.json、model.safetensors |

真实加载后计数：完整模型 **159,647,232** 参数，视觉编码器 **94,552,320** 参数，非视觉部分 **65,094,912** 参数。V1 只对非视觉部分做过直接计数；本次补齐了视觉与整机精确统计。

在仓库自带的两张公开演示图上，每张分别运行 64、16、4 三种预算，共六次生成；均使用同一 checkpoint 和 greedy decoding，输出最多 24 个 token。六次 logits 都有限，均在 24-token 上限停止而非 EOS，因此报告保存的是回答前缀，不是完整答案。

第一张图在 64 档的前缀描述“金毛寻回犬”，16 档描述“狗”，4 档改成“飞机”。第二张图三个预算都输出以“雨伞”为主的前缀。这是可追溯的压缩敏感性实例；两张演示图不是独立 benchmark，不能据此统计泛化准确率，更不能声称某个预算无损。

完整原始观测与参数、环境记录见 [BUDGET_V2_SMOKE.json](evidence/BUDGET_V2_SMOKE.json)。脚本为 `scripts/smoke_pretrained_budget.py`，没有对生成结果做人工正确性评分。

## 真实权重：已完成一次 64→16 蒸馏更新和重载

使用上述两张图的 64-token 教师生成前缀作为伪标签构造 Parquet。伪标签未经过人工审核，不是 gold answer；24-token 截断前缀尤其不应被当作正式训练数据。这个小集合只用来测试数据、双路前向、反向、优化器、保存与重载。

实际运行：CPU float32，batch size 1，累积设置为 4，但数据只有 2 个微批；训练 Projector 与首末层，学习率 `5e-6`，只执行 1 个 optimizer update。

```text
有效答案 token：58
CE 均值：1.2816878023
温度缩放后的 KL 均值：0.5358335890
CE + KL：1.8175213912
学生 padding 后位置总数：158
完整预算源序列位置总数：254
```

158 与 254 是该次输入形状统计，不是推理速度或训练 FLOPs 的实测节省。日志的 teacher next-token accuracy 是参考前缀下的 token 命中率，不是自由生成问答准确率。

更新后的学生已保存到本地忽略目录，并通过 native loader 根据 sidecar 重新加载到 16-token 配置；真实图片前向 logits 有限。Projector 一层权重相对初始化的最大绝对更新为 `5.036592483520508e-06`。这证明权重实际更新且可以重新加载，**不证明蒸馏提升了模型能力**。未将这份仅训练一步的权重作为效果模型发布或上传 GitHub。

复现步骤（下载好固定版本资源后，从仓库根目录运行；输出目录必须不存在）：

```bash
python scripts/smoke_pretrained_budget.py \
  --output_dir results/v2_pretrained_probe --max_new_tokens 24 --threads 4

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python trainer/train_budget_vlm.py \
  --data_path results/v2_pretrained_probe/smoke_pseudolabels.parquet \
  --init_checkpoint out/sft_vlm_768.pth \
  --teacher_checkpoint out/sft_vlm_768.pth \
  --run_dir results/v2_real_distill_smoke \
  --image_token_len 16 --batch_size 1 --accumulation_steps 4 \
  --max_updates 1 --max_seq_len 256 --freeze_llm 1 --device cpu
```

## 仍然没有完成

没有独立领域任务训练/评测、多训练 seed 的准确率对比、CUDA AMP 或 NCCL VLM 多卡训练、GPU 延迟与显存对比、移动端部署、vLLM/SGLang 原生适配、on-policy 蒸馏、3D 特权输入或自动局部放大。本版不支持精确中断续训。动态形状在不同 attention 路径下可能有不同开销，不能把 token 数下降直接写成部署加速比例。

GitHub Actions 工作流会对提交的文件运行 CPU 测试；远程状态以当前 commit 的 Actions 结果为准。本记录的真实权重探针需要另行下载权重，不在日常 CI 中重复下载和运行。
