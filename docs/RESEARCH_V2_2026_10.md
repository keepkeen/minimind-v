# V2 研究更新：哪些真正能迁移到小 VLM

核查日期：**2026-10-09**。论文日期按 arXiv 首次提交/修订分别列出；10 月 9 日公告中的最新文章可能在 10 月 8 日提交。以下主要是一手预印本与官方工程文档，不把新论文视作共识，不把作者 benchmark 数字移植为本项目成果。

## 已转化为代码的最小主线

### LT-OPD：2026-09-26 首版，2026-10-01 修订

[Fewer Tokens, More Self-Teaching](https://arxiv.org/abs/2609.32353)研究极低视觉预算下的 on-policy 自蒸馏：学生用压缩输入生成轨迹，冻结的全预算同源教师在学生生成前缀上提供分布监督，并逐步降低预算。

本项目据此把“重新学习使用压缩表示”作为研究假设，先实现同源 64-token 教师与 16/4-token 学生的可控训练入口。**当前是固定参考答案前缀的 offline KL**，未实现 rollout、预算课程与该论文的完整目标；不能写成已经复现 LT-OPD。

### SCOPD：2026-09-28

[SCOPD](https://arxiv.org/abs/2609.34044)的观察是：固定裁剪视觉表示上的多次采样，可以找回部分 greedy 错题；因此性能损失不一定都来自不可逆信息丢失。论文采用 on-policy 稀疏上下文自蒸馏，SCOPD+ 再利用预算干预筛选视觉敏感回答位置。

本项目保留正确图/黑图/错图诊断，并加入逐题配对、同图分组统计。这个评测框架为后续区分“看不到”和“不会用”提供数据基础，但没有实现论文 Pass@K 分析与视觉敏感位置蒸馏。不能以某个词发生变化单独证明视觉证据被正确利用。

### TBD：2026-08-28

[Token-Budget Distillation](https://arxiv.org/abs/2608.28138)更接近本版离线起点：视频学生使用压缩 token，全预算教师提供答案区域 KL，配合任务损失、GT margin 和可靠性控制。原论文使用 LoRA、FlashVID 与视频骨干。

本版只采用“全预算教师监督低预算学生”的公共设计，并实现 CE + answer KL 的最小对照。没有移植 LoRA、FlashVID、视频、margin loss 或教师门控。特别处理了预算导致的答案位置错位，逐样本检查师生监督标签序列一致，而不以截取最短长度隐藏错配。

## 值得接着实验，但尚未实现

### KVE-KD：2026-10-02

[Key Visual Evidence-Guided Knowledge Distillation](https://arxiv.org/abs/2610.03842)以生成前的最后一个文本 token 为语义锚点，通过视觉贡献干预选择融合层，再利用锚点条件注意力与熵选择关键视觉 token 做特征蒸馏。

它针对“所有图像 token 等权监督会混入背景”的问题。本项目可以在 CE+KL 稳定后，比较额外的关键视觉特征监督与同等数量随机/均匀区域监督。要先增加可审计的中间层输出，并检查额外显存与计算；不能直接把某个 attention 热图当作因果解释，也不宜一次叠加多个未经比较的 loss。

### 小目标与局部放大：2026-10-07

[Why VLMs Miss Small Objects, and When Zooming In Is Safe](https://arxiv.org/abs/2610.09313)从对象跨越的 token 数与每次调用覆盖内容量分析可见性和覆盖成本，在一定条件下讨论局部分解的召回边界。

对于 MiniMind 固定 256 图像输入，这比盲目增加文本推理长度更相关。候选实验是原始高分辨率全图+局部裁剪，并分别评估局部读取、全局计数和跨区域空间关系。上游已缩小到 256 的 JPEG 不能恢复原始细节；裁剪视图的额外 ViT 和 token 成本必须计入。当前未加入自动 zoom agent，也没有高分辨率训练数据。

### 3D 特权蒸馏 GPD：2026-10-08

[Distilling Routed 3D Privilege for Spatial Reasoning in VLMs](https://arxiv.org/abs/2610.12355)将深度、语义和鸟瞰几何信息按问题路由给教师，结合错误轨迹上的 privileged KL 与 GRPO，部署模型仍然只接收 RGB。

这可作为未来连接 3D/空间理解经验的方向：训练期教师看到更多真实几何，学生推理期无需额外传感器。但需要几何标注与独立空间任务集；当前 64-token 图像教师不具备 3D 特权，代码也没有实现该论文的 GRPO 或几何路由。

## 业界实现：区分缓存层，先找真实瓶颈

### vLLM 多模态处理文档：页面标注 2026-10-05

[官方设计文档](https://docs.vllm.ai/en/latest/design/mm_processing/)说明了图像占位位置与多模态输入的严格对应、processor 输出缓存，以及将 uint8 图像传输到 GPU 后融合 normalize/rescale 的优化路径。

Processor 缓存节省预处理，不等于省掉 ViT。GPU 归一化也仍保留 CPU 解码与 resize；它只适用于文档列出的受支持模型路径。本项目没有获得 vLLM 原生适配，不能因为继承 Transformers 类就声称已经支持 vLLM continuous batching 或 CUDA graph。

可迁移的工程原则是把预处理、ViT、Projector、prefill、decode 分开测。V2 共享冻结 ViT 的原始特征，并在 CLI 记录配置，是这种分层分析的一部分；GPU fused normalization 暂不实现，因为这里尚无对应硬件瓶颈测量。

### SGLang 编码器缓存：截至 2026-10-09 的官方接口

[Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)中 `--enable-mm-global-cache` 为 encoder server 提供 Mooncake-backed 全局多模态 embedding 缓存，以重复使用 ViT 输出。

这与预处理缓存、语言解码器 KV cache 是三个不同层次。本项目目前只共享微批内冻结视觉特征，没有跨服务器缓存。实际产品同图多问时可进一步测缓存命中率、失效条件、显存预算与租户隔离；不是先加一个字典就宣称达到了生产级分布式缓存。

### 梯度累积正确性：经典工程实践，不是 2026 创新

[Hugging Face 官方说明](https://huggingface.co/blog/gradient_accumulation)发表于 **2024-10-16**，指出 token 级损失应该按累积窗口所有非 padding token 统一归一化。本次将这一原则落实到 CE/KL、尾窗口和 DDP rank，并用两进程 CPU 测试验证梯度一致。

这里的研究增量是设计可检验的低预算适配实验，工程增量是让输入、损失与评测正确。两种贡献分开报告，有助于避免把训练 bug 的修复误写成一种新蒸馏算法。

## 推荐优先级

先完成 64 CE / 16 CE / 16 CE+KL 的同数据、多 seed、独立测试集比较。下一步按错误类型选择：若是生成偏离但证据仍在，研究 LT-OPD/SCOPD；若是小目标不可见，研究原始图像分辨率与裁剪；若是几何关系缺失，考虑有真实 3D 标注的教师。只有确定瓶颈后再接量化、专用推理引擎或新的辅助目标。
