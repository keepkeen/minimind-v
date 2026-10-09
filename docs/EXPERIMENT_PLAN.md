# BudgetLab：实验与运行手册

## 目标与当前边界

目标是建立“任务效果—视觉信息预算—端到端成本”的可复现实验，而非给通用聊天 Demo 换名字。候选落地任务是现场图像中的受限视觉问答，例如大对象存在性、柜门开闭、指示灯颜色等；输出为少量字段或短答案，必须保留人工复核。复杂小字 OCR、医疗结论和自动设备控制不作为首期承诺。

业务价值仍需证明：同一离线设备上，是否存在一个足够准确、足够便宜的方案。若专用分类器、OCR+规则或 0.8B 级 VLM 在约束下更合适，应接受这一结果。

已经实现的是实验基础设施。尚未完成真实任务训练和质量对比，下面的 H1/H2 都是假设。

## 可区分的假设

H1：对依赖大尺度外观的短答案问题，16-token 池化在部分任务上可能保留足够信息并降低 prefill 成本。反例机制是空间混合损伤细节，或冻结视觉编码器成为主要瓶颈，从而效果下降却没有可用的端到端收益。

验证 H1 要同时测准确率、TTFT、完整请求耗时、输出长度和峰值显存。分别列出颜色、存在性、计数、位置、小字切片。4-token 是压力测试，不预设可部署。

H2：同一图片重复提问时，复用冻结/已对齐的图像特征能够避免重复编码，且在同权重、同处理器和 eval 模式下保持 logits 一致。已有合成模型数值测试验证接口；真实设备收益尚需测量。缓存实验与 token 压缩实验分开，避免把两个改动混成一个“算法增益”。

## 数据与对照设计

按图片、文档或采集场景分组拆 train/dev/test；相同图像的中英问答和相邻视频帧不得随机散到不同集合。保留原始图像、来源、许可证、答案和任务类别；需要小字能力时不能只使用上游已缩到 256 的 Parquet。

第一组实验：固定已有 64-token checkpoint，推理期分别切换 64/16/4，观察分布变化的直接损伤。结论仅适用于推理期压缩，不称为重新训练后的能力。

第二组实验：从同一语言 checkpoint、同一 projector 随机初始化及相同数据顺序分别训练 64/16/4。开发阶段可以单种子；正式结论至少报告预先固定的多种子和按图片分组的 bootstrap 区间。默认脚本 seed 固定，要做多种子需明确记录改动或另加 seed 参数。

固定回答文本的有效监督预算。当前数据集 padding 到固定 max_length，因此 token 数变少并不自动使训练张量变短，也不保证训练更快；它可能仅让更多文本保留到截断长度内。若比较训练吞吐，应另行实现并验收动态 padding/长度分桶，报告实际输入长度，不能把预期推理收益当训练收益。

对照包括：上游 64-token（原实现）、修复后 64-token、16-token、4-token；应用层额外比较任务适合的分类器或 OCR+规则，以及 Qwen3.5-0.8B、MiniCPM-V4.6 等外部 VLM。外部模型的规模、视觉分辨率、训练数据不同，不是纯算法消融。

## 建立测试环境

本次测试使用 Python 3.12。不要直接假设上游旧依赖可用于 Python 3.14。

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
# CPU / macOS；CUDA 环境按自己的驱动与 PyTorch 官方安装方式选择 wheel。
python -m pip install torch==2.6.0
python -m pip install -r requirements-test.txt
python -m pytest -q tests
```

完整训练还需要模型权重、视觉编码器和 Parquet 数据；下载位置与原版 README 一致。本次没有自动下载训练数据、购买算力或启动长时间训练。

## 训练：明确目录和预算

下列命令是可执行入口，不是已完成实验。先准备 `out/llm_768.pth`、`model/siglip2-base-p32-256-ve/` 与 `dataset/sft_i2t.parquet`。

```bash
cd trainer
# 直接 SFT。每个预算使用独立输出名，防止权重混淆。
python train_sft_vlm.py --from_weight llm --freeze_llm 1 \
  --image_token_len 64 --save_weight sft_b64 --epochs 2
python train_sft_vlm.py --from_weight llm --freeze_llm 1 \
  --image_token_len 16 --save_weight sft_b16 --epochs 2
python train_sft_vlm.py --from_weight llm --freeze_llm 1 \
  --image_token_len 4 --save_weight sft_b4 --epochs 2
cd ..
```

可选对齐阶段同样支持 `--image_token_len`。16 档示例：`cd trainer` 后先用 `train_pretrain_vlm.py --from_weight llm --image_token_len 16 --save_weight pretrain_b16`，再从 `pretrain_b16` 进行同预算 SFT。

恢复 checkpoint 会验证 `image_token_len`。普通原生 `.pth` 权重文件仍不自带完整配置，因此必须同时保存训练命令、预算、模型配置与数据版本；不要只凭文件名猜参数。HF 导出时也必须使用正确 VLMConfig。

## 评测清单格式

UTF-8 JSONL，每行一个问题；图片路径相对于清单文件。以下只有格式示例，文件和答案需替换为自己的标注，不能当作已经创建的数据。

```json
{"id":"panel-001-color","image":"images/panel-001.jpg","question":"指示灯是什么颜色？只输出颜色。","answers":["红色","红"],"category":"indicator_color"}
{"id":"pair-002","images":["images/a.jpg","images/b.jpg"],"question":"<image>\n<image>\n两张图片中柜门是否都关闭？只回答是或否。","answers":["是"],"category":"multi_image_state"}
```

问题不含 `<image>` 时，脚本会按图像数量在开头添加占位符；已经包含时必须数量相等。不会静默截断图像位置。`id` 必须唯一，answers 是非空字符串列表。类别应用稳定的标签命名。

```bash
# 从仓库根目录运行；先准备真实权重和清单。
python scripts/eval_budget_vlm.py --manifest dataset/holdout.jsonl \
  --weight sft_b64 --image_token_len 64 --device cuda \
  --output results/b64_actual.jsonl
python scripts/eval_budget_vlm.py --manifest dataset/holdout.jsonl \
  --weight sft_b16 --image_token_len 16 --device cuda \
  --output results/b16_actual.jsonl
python scripts/eval_budget_vlm.py --manifest dataset/holdout.jsonl \
  --weight sft_b16 --image_token_len 16 --device cuda --image_condition blank \
  --output results/b16_blank.jsonl
# shuffle 当前仅支持单图清单，按不同图片身份确定性错配。
python scripts/eval_budget_vlm.py --manifest dataset/single_image_holdout.jsonl \
  --weight sft_b16 --image_token_len 16 --device cuda --image_condition shuffle \
  --output results/b16_shuffle.jsonl
```

运行后生成逐样本 JSONL 和同名 `.summary.json`。输出路径存在时拒绝覆盖。实际图和错图条件必须使用相同问题、权重与生成配置；干预结果不单独作为“无幻觉证明”。

## 指标边界

本地 normalized EM 只做 NFKC、大小写和空白归一化，保留小数点、符号与数字。ANLS-style 为多参考答案的最大归一化编辑相似度，距离达到 0.5 则记零。它们不是官方 VQAv2、DocVQA 或 POPE 实现；正式宣称榜单结果前须接官方 evaluator。

计时固定 batch=1、greedy、默认 warmup=1。TTFT 从预处理完成后开始，包含视觉编码与 prefill；generation_ms 包括完整生成；request_ms 额外包含图片 IO/预处理，不包含模型加载与答案 detokenization。decode_tps_after_first_token 单独估算首 token 之后的吞吐，生成不到两个 token 时为 null。计时包含 streamer 的观测开销。

CUDA 报告 allocated/reserved 峰值，包含模型驻留内存；CPU/MPS 这两个字段为 null，不能把 null 写成显存为零。图片大小、题目长度、答案长度、线程数、设备和精度都会影响比较。报告环境版本、git commit 和 dirty worktree 标志；长时间服务还需要并发、p95、缓存命中率与能耗测试，本脚本不冒充生产压测框架。

## 何时才算有价值

先定义目标设备与验收阈值，再选满足约束的最小成本配置。可采用“字段准确率与基线差不超过预先约定容忍值，同时 p95 请求耗时符合业务要求”作为开发标准；具体阈值由真实业务确定，不先写成已实现数字。

如果 16 档伤害关键细节，保留 64 档，或用高分辨率原图重新设计输入。若视觉编码占主导，优化编码器或同图缓存比继续压 decoder token 更值得。若语言容量限制回答，则考虑短答案蒸馏或较大主干。每一步都对应可测现象，而不是叠加模块。

后续可探索学习式重采样、任务相关 token 选择、短结构化教师标签和选择性拒答。它们尚未实现，也没有新颖性或效果结论。
