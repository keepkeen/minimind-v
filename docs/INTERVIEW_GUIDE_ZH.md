# MiniMind-V：源码讲解与面试手册

审阅日期：2026-10-09。上游基线：`1862b633fc082a723e78dbead9777545f09d960c`。
本文区分上游已有能力、BudgetLab 本次代码贡献，以及尚未完成的训练实验。

## 1. 先把项目说准

MiniMind-V 是把已训练的 MiniMind 语言模型与冻结视觉编码器连接起来的小型视觉语言模型。当前版本是 MiniMind-3V，不是早期 CLIP 版 MiniMind2-V。它采用 SigLIP2 P32 视觉编码器、LayerNorm + 两层 MLP 投影和自回归语言解码器。

上游“从零”强调自己实现 VLM 和初始化连接层，不表示本仓库从随机参数训练了视觉编码器和语言主干。“65M”也不是推理整机参数量。按默认 dense 配置直接计数：LLM 63,912,192；projector 1,182,720；两者合计 65,094,912。上游另报告冻结视觉编码器约 95M，因此完整推理系统约 160M。默认 SFT 只更新 projector 和第一、最后一个 Transformer block，实际可训练参数为 15,931,776。

因此至少报告四种口径：总参数、非视觉参数、可训练参数、MoE 每 token 激活参数。冻结不等于删除；MoE 激活少不等于所有专家不占显存。

上游的低价复现口径针对其指定 GPU、数据和训练阶段，不包含获得语言与视觉预训练能力的全部成本。租价、吞吐与复现时长也不应作为你的实测成绩。

来源：[上游 README](https://github.com/jingyaogong/minimind-v/blob/1862b633fc082a723e78dbead9777545f09d960c/README.md)、[VLM 实现](https://github.com/jingyaogong/minimind-v/blob/1862b633fc082a723e78dbead9777545f09d960c/model/model_vlm.py)、[LLM 实现](https://github.com/jingyaogong/minimind-v/blob/1862b633fc082a723e78dbead9777545f09d960c/model/model_minimind.py)。

## 2. 代码地图与阅读顺序

| 文件 | 要回答的问题 |
|---|---|
| `model/model_minimind.py` | 文本 Transformer 怎样计算注意力、位置编码、FFN、MoE 和 KV cache？ |
| `model/model_vlm.py` | 图像如何编码、投影、替换文本占位符，以及如何走自回归生成？ |
| `dataset/lm_dataset.py` | Parquet 怎样变成对话、图像张量和只监督 assistant 的标签？ |
| `trainer/trainer_utils.py` | 冻结策略、参数统计、组 batch、恢复训练状态如何实现？ |
| `trainer/train_pretrain_vlm.py` | 仅训练连接层的对齐阶段如何运行？ |
| `trainer/train_sft_vlm.py` | 指令微调时更新哪些参数？ |
| `eval_vlm.py` | 原生权重和 HF 权重怎样加载、怎样做示例生成？ |
| `scripts/convert_vlm.py` | 怎么导出模型，为什么还需要单独的视觉编码器？ |
| `scripts/web_demo_vlm.py` | UI 怎样调用模型？它不负责证明模型能力。 |

BudgetLab 新增 `scripts/eval_budget_vlm.py`、`scripts/vlm_metrics.py`、`tests/` 与本组文档。面试可以先画前向数据流，再对照这些入口定位。

## 3. 一张图的完整前向过程

默认 dense、单图情况下：

```text
RGB 图像
  -> SiglipImageProcessor：固定 256×256
  -> pixels [B, 3, 256, 256]
  -> 冻结 SigLIP2 ViT-B/32
  -> patch features [B, 64, 768]
  -> LayerNorm -> Linear(768,768) -> GELU -> Linear(768,768)
  -> visual embeddings [B, 64, 768]

对话文本含 <image>
  -> <image> 展开为 64 个 <|image_pad|>
  -> input_ids [B,L]，token embedding [B,L,768]
  -> 对应 64 个位置替换成 visual embeddings
  -> 8 个因果 Transformer block
  -> RMSNorm -> tied LM head
  -> logits [B,L,6400]
  -> 逐 token 输出答案
```

64 来自 `(256 / 32)^2 = 8×8`。本实现取 `last_hidden_state` 的 patch 表征，不把图像先变成一串 OCR 文本，不使用单一全局图像向量作为全部输入，也没有训练 SigLIP 的图文对比损失。

README 部分措辞提到 NaFlex，但当前加载的是 `SiglipVisionModel` 与固定尺寸 `SiglipImageProcessor`。面试应按实现称为固定 256 输入路径，不能把它讲成已经实现原生动态分辨率或任意分辨率图像切块。

这里的“融合”是 embedding 级替换，图像与文本随后使用同一解码器的自注意力。它没有额外的独立 cross-attention 层。图像位置仍沿用该解码器的一维 RoPE；ViT 先在图像内部处理空间位置。不要声称它已有 Qwen-VL 式专门的多维位置编码。

两层 MLP 对每个视觉 token 做非线性空间映射。它本身不会改变 token 数，改变的是特征表示。本 fork 新增的空间平均池化放在 MLP 之前：8×8 可以变为 4×4 或 2×2，随后分别得到 16 或 4 个 token。64 档直接返回原特征，不引入新可学习参数。

## 4. 文本主干必须能讲到的细节

默认 `hidden_size=768`、8 层、8 个 query heads、4 个 KV heads，head_dim=96，词表 6400。GQA 让两组 query heads 共享一组 K/V，减少 KV cache；它没有把 query head 数也压成 4。

每层为 pre-norm 结构：`x + Attention(RMSNorm(x))`，然后 `x + FFN(RMSNorm(x))`。Q、K 还有独立 RMSNorm。FFN 采用 SwiGLU：`down(SiLU(gate(x)) * up(x))`；默认中间维度为 2432。词嵌入和输出头共享权重，参数统计不能重复算两份。

RoPE 把位置信息编码进 Q/K 的旋转关系。配置允许较长位置范围，但“预计算了 32768 位置”不等于模型已被验证具有可靠 32K 理解能力。训练长度与数据覆盖仍然决定有效上下文。

代码调用 PyTorch `scaled_dot_product_attention` 的路径受 mask、cache 与序列长度条件限制。使用这个接口不代表所有设备、dtype 与输入都会实际运行 FlashAttention 内核。

可选 MoE 有 4 个专家，默认 top-1 路由，并使用负载均衡辅助损失。应区分增加模型容量与增加每 token 激活计算；也要考虑专家负载、路由训练、通信和实际内核效率。该小型实现不是大型生产 MoE 系统的完整缩小版。

## 5. 训练目标、mask 和反向传播

目标是给定图像、问题和已有答案前缀，预测下一个答案 token：

`L = - sum_t m_t log p(y_t | image, question, y_<t) / sum_t m_t`。

实际源码先把不监督的位置设成 `-100`，再对 `logits[..., :-1, :]` 与 `labels[..., 1:]` 做交叉熵。MoE 额外加路由辅助项。user、system、图像占位位置和 padding 不应直接承担答案生成的 CE 目标。

“labels 被 mask”与“attention 被 mask”是两件事。图像和问题虽然不参与输出损失，答案仍然可以注意它们；不训练预测问题不表示把问题从模型里删掉。当前训练使用右 padding，正常答案位置在因果 mask 下不会看到后面的 pad。未来若做左 padding、packing 或改 attention 路径，必须重新处理 attention mask 与位置。

冻结视觉编码器时可对视觉前向使用 `torch.no_grad()`，因为不需要对它更新。冻结 LLM 时却不能把整个 LLM 前向放进 `no_grad()`：答案损失仍须经 LLM 的输入梯度回到 projector。`requires_grad=False` 阻止参数梯度，通常不阻止对输入的梯度。

## 6. 当前训练配方不是机械的“两阶段都必跑”

| 阶段 | 默认学习率 | batch | 最大长度 | 默认冻结策略 |
|---|---:|---:|---:|---|
| Pretrain | 4e-4 | 16 | 450 | `freeze_llm=2`：仅 projector |
| SFT | 5e-6 | 4 | 768 | `freeze_llm=1`：projector + 首尾两块 |

两者默认 epochs=2、梯度累积=1。以当前脚本默认值为准，而不是混用 README 中别的版本实验参数。

`freeze_llm=0` 解冻全部非视觉参数；1 只解冻第一块和最后一块；2 只训练 projector。第一块负责初步处理新模态、最后一块直接影响输出，是一个可理解的低成本启发式；这不是已证明的最优策略，必须用验证集和消融比较全量微调、仅 projector 与其他选层方案。

上游 SFT 文件已包含 caption 子集，所以 README 推荐从 `llm` 直接 SFT。对齐 pretrain 仍可以作为额外实验，但不要说它在当前复现路径中强制必需。

混合精度、梯度裁剪、AdamW、DDP、余弦学习率与 checkpoint 都在原生 PyTorch 脚本内完成。bf16 与 fp16 的指数范围不同；代码仅为 fp16 启用梯度缩放。DDP 复制模型到每个设备，不自动把模型参数分片。学习率是衰减到初始值的 0.1 倍的余弦形式，没有独立 warmup。

恢复训练会保存模型、优化器、scaler、epoch/step 等，但不能直接称为逐位一致恢复；原代码没有完整恢复所有随机状态或中间梯度累积状态。多卡、CUDA AMP、torch.compile 和大数据长跑都需要单独验收。本次没有完成这些验收。

来源：[上游训练目录](https://github.com/jingyaogong/minimind-v/tree/1862b633fc082a723e78dbead9777545f09d960c/trainer)。

## 7. 数据不是附属环节

上游 README 报告 pretrain 约 127 万条、SFT 约 290 万条，来源为 ALLaVA-4V 系列及增广数据。SFT 包含 caption 子集和约 23 万条纯文本对话。样本数不等于唯一图片数；不同语言、不同问题可能共享同一张图。

因此不能随机按对话行拆训练和测试。应按原始图片 ID、文档 ID 或采集场景分组，再划分数据；否则同图中英问答或不同 caption 会跨集合泄漏。工业任务还应把同一设备、拍摄序列和模板变体的近重复一起分组。

图像已被统一缩到 256×256 并封入 Parquet。丢掉的小字细节不能通过给这张小图再做切块恢复。高分辨率实验需要重新取得原始图像，并把多视图数量、视觉 token、预处理与编码成本全部计入。

本 fork 在数据链路增加图片数与占位符计数校验、截断校验、无监督答案拒绝、padding 标签屏蔽，以及变长多图批次支持。纯文本样本不再编码上游填充的黑色占位图。这些首先是正确性和资源使用修复，不是准确率提升证据。

## 8. 推理成本与缓存

分清三个阶段：图像预处理和编码，问题与图像的 prefill，答案的逐步 decode。上游示例程序把总生成耗时除以生成 token 数，其数值混合了视觉编码、prefill 和 decode，不应当作纯 decode 吞吐。

prefill 时，图像进入网络并填充每层 KV cache；后续 token 基于 cache 推理，不应重新编码同一图。跨问题复用图像则是另一层缓存：本 fork 提供 `encode_images()` 和 `image_embeddings=`，可以复用投影后的图像特征，但新的问题仍需要新的文本 prefill。

```python
model.eval()
with torch.inference_mode():
    features = model.encode_images(pixel_values)
    output = model.generate(
        inputs=input_ids, image_embeddings=features,
        do_sample=False, temperature=1.0, top_k=0, top_p=1.0,
        max_new_tokens=64,
    )
```

修改 checkpoint、视觉预处理、token 预算、dtype/device 或图片后，应使相应缓存失效。不要在 projector 更新后继续使用旧投影特征训练。

对于当前 dense 的 KV cache，单图 64 token、8 层、4 个 KV heads、head_dim96、fp16 的理论视觉部分约为 `2×8×64×4×96×2 = 786432` 字节，即 0.75 MiB；这不包含文本 cache、模型参数和临时激活。

64→16 让视觉序列缩短 75%，但不会让系统延迟自动降低 75%。例如文本长度 100 时，总输入从 164 变为 116；注意力矩阵大小约变为 `(116/164)^2`，FFN token 工作量约变为 `116/164`，视觉编码器工作量却没有降低。最终必须测墙钟时间、峰值显存和任务效果。

## 9. 本次代码贡献与证据边界

本次新增 64/16/4 空间池化预算、严格视觉融合、变长多图与纯文本混合批次、跨问题视觉特征缓存入口、数据截断与监督校验、完整参数统计、确定性 QA 评测及测试。

对上游相邻图片问题做了最小复现：输入 132 个位置，其中有连续两张图的 128 个 marker，上游函数输出长度变成 68；修复后仍是 132。原因是原函数把连续 marker 的整段当成一张图，截取第一张图的 64 个特征后缩短了序列。新实现按所有 marker 的顺序写入展平后的有效视觉特征，并校验数量恰好匹配。

默认 64 档的 projector 运算与权重 key 保持兼容。16/4 档是空间平均池化基线，不是 P2P、MiniCPM 混合压缩或一种新发表的注意力选择算法。不能把它宣传成“无损 16 倍加速”。

截至本次提交，CPU 合成模型/真实 tokenizer/Parquet/评测入口测试通过。没有完成完整预训练、SFT、真实权重任务基准、CUDA 训练、多卡或设备端性能实测。详见 [验证记录](VALIDATION.md)。

## 10. 高频面试追问

**为什么不用 OCR 再交给 LLM？** OCR 链路对文字抽取强、可解释，但会丢失非文字物体和部分布局关系。端到端 VLM 能利用视觉信号，但小模型的识字精度不一定更好。应按任务比较 OCR+规则、OCR+LLM 与 VLM，不能预设 VLM 必胜。

**为什么使用 patch 特征而非全局向量？** 空间细节仍留在多个视觉位置中，有利于局部问答；代价是输入长度和计算。把图像压成一个向量更便宜，但可能损失小对象和关系信息。

**MLP projector 与 Q-Former/resampler 有何区别？** MLP 逐 token 映射，结构简单；查询型重采样器可通过可学习 query 汇总可变输入并控制输出长度，通常增加训练与注意力开销。平均池化是一条可解释的廉价对照线，不具备问题相关选择能力。

**为什么先冻住视觉编码器？** 重用其预训练表征、降低反向与优化器成本、减少小数据下破坏表征的风险。但仍有前向成本，领域迁移严重时完全冻结可能限制效果。

**只训练首尾层是不是最新论文证明的？** 不是。参数更新价值、激活重要性与可跳过的推理计算不是同一个问题。不能用别的模型上的层级消融为本项目冻结策略直接背书。

**如何证明模型真的看图？** 配对比较正确图、黑图与错误图，并按视觉必需问题分别计分。答案变化只能说明敏感性；应比较正确性变化，且黑图本身是分布外干预。进一步需要最小图像编辑对、区域证据和人工检查。

**能直接把 64 档 checkpoint 改成 4 档当提升吗？** 只能把它作为推理期分布变化实验。更完整的比较是从相同初始化分别训练 64/16/4，再在同一测试集报告成本—效果。checkpoint 文件名与恢复 metadata 要区分预算。

**还应该增加多少层 CoT 或马上跑 GRPO？** 先证明基础感知能解题，再考虑短结构化推理、教师筛选和可验证奖励。小主干可能连关键文字都看不清，增加长文本推理无法恢复丢失像素。未验证奖励与基础能力时，RL 可能只是强化猜测或奖励漏洞。

**怎么做困难样本？** 小目标、细字、反光、遮挡、相近颜色、计数和空间关系分别建切片；用确定性的匹配题和近邻错误图，记录失败类型，避免只展示容易的自然图 caption。

**为什么仅训练 loss 不够？** 模型可能利用语言先验、重复训练图或答案模板降损失。必须在独立图片/场景测试集上检查精确字段、错误图干预和资源成本。

**这个项目什么时候值得放简历？** 能讲清源代码、复现具体 bug、展示你写的测试和实验记录时，可以作为扎实的工程项目。要写算法效果，需要实际跑完受控比较。fork 原作者代码本身不能作为原创算法成果。

## 11. 可使用的项目陈述

当前事实版本：“基于 MiniMind-3V 构建面向受限算力的视觉问答实验平台，完成可配置视觉 token 预算、严格多图融合、特征缓存和确定性评测；复现并修复相邻图像导致序列缩短的问题，补充数据与模型回归测试。正在建立按任务切片的成本—效果实验。”

完成真实实验后再补充：“在某设备、某精度、某独立测试集上，16-token 方案相对 64-token 基线的字段 EM 为 X→Y，TTFT 为 A→B，峰值显存为 C→D；报告三种种子及不确定区间。细字切片退化，因此对该类任务保留高预算。”X/Y/A/B/C/D 必须来自保存的实验产物。

上游工作保留署名和许可证。代码、权重和数据来源分别记录；不能用代码开源许可证替代数据授权审查。
