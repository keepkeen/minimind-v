# BudgetLab V2：从压缩接口到可控训练实验

日期：2026-10-09。V1 基线：`4b4201177cb8e7bb2facc44ef1891a7639ab7aa4`。

本版增加可实际运行的 **offline teacher-forced CE + forward-KL 蒸馏**。教师是冻结的 64-token VLM，学生使用 16 或 4 个视觉 token；两者使用相同词表、问题和答案前缀。它是一个可解释基线，**不是 LT-OPD/SCOPD 的 on-policy 复现，也不是完整 TBD**。关联论文与业界实现见 [研究更新](RESEARCH_V2_2026_10.md)，执行证据见 [V2 验证记录](VALIDATION_V2.md)。

## 1. 为什么先做这些改动

V1 的空间池化只减少视觉输入位置，没有使压缩后的模型重新适应表示变化；原数据集还将每条样本 padding 到固定上限，因此没有自动缩短训练张量。

另一个先复现再修复的问题是梯度累积：当微批包含的有效答案 token 数不同，平均微批 loss 不等于全体答案 token 的平均 loss；最后一个累积窗口不足设定步数时，固定除数还会缩小梯度。最小两类实验中，1-token 与 3-token 两个微批的旧梯度与整体 token 均值梯度相差 0.25。

这些属于工程正确性问题。它们解决后，才有意义比较蒸馏是否改善低预算模型。新实验使用 `trainer/train_budget_vlm.py`；原来的 `train_pretrain_vlm.py` 和 `train_sft_vlm.py` 保留为历史复现入口，其旧累积逻辑并未被本次全量替换。

## 2. 数据与损失的精确约定

训练数据先展开成 64-token 图像占位序列，再按 `max_seq_len` 截断。学生仅压缩图像占位符，保留所有非图像 token 和 labels。因此 CE-only、CE+KD、64/16/4 的对照不会因截断位置变化而获得不同的答案文本。

默认按批内最长样本动态右 padding，并由真实长度构造 attention mask；不以 `token == pad_id` 推断长度，避免 pad 与 EOS 共享 ID 时出错。`--padding fixed` 可恢复固定形状进行独立性能对照。此实现没有跨样本 packing；它不会把不同样本串接后让答案跨样本注意。

损失为：

\[
L=\frac{1}{N}\sum_{i,t\in A_i}\left[-\log p_S(y_{it}\mid I_b,x_i,y_{i,<t})
+\lambda\tau^2\mathrm{KL}\left(p_T^{(\tau)}(\cdot\mid I_{64},x_i,y_{i,<t})\;\|\;p_S^{(\tau)}(\cdot\mid I_b,x_i,y_{i,<t})\right)\right].
\]

`A_i` 是被监督的 next-answer-token 位置，`N` 是整个梯度累积窗口、全部 DDP rank 的有效答案 token 总数。图像、用户问题和 padding 不直接参与 CE/KL。默认 `tau=2`、`lambda=1` 只是起点，不是经调参证明的最优值。

图像预算变化会移动答案的绝对位置。`answer_kl_sum` 逐样本按监督 token 顺序对齐，并强制检查师生监督 token 序列完全相同；不能截取两个 logits 张量的相同绝对区间，也不能取最短长度来掩盖错位。

教师分布来自固定参考答案前缀，即 teacher forcing。没有学生 rollout，没有 GRPO，没有教师置信度门控，也没有长 CoT 生成或跨词表蒸馏。`teacher_next_token_accuracy` 是参考前缀下的 token 命中率，**不是教师自由生成的问答准确率，更不能代替任务指标**。

## 3. 共享视觉计算、累积与配置

`encode_image_features` 提供冻结编码器的 `[B,N,64,Dv]` 原始 patch 特征。教师与学生共享这一次 ViT 结果，分别执行各自的 Projector 与语言主干。学生 Projector 在 `forward` 内运行，保留 DDP 管理的可训练计算图。

这与 V1 的投影后 `image_embeddings` 缓存不同：Projector 更新会使投影后缓存失效；冻结编码器和预处理未变化时，原始 patch 特征可以复用。当前仅在同一训练微批共享，不实现磁盘特征仓库、跨请求 LRU、跨服务器缓存或自动缓存失效服务。

教师仍然增加一次语言主干前向及其内存成本。共享 ViT 不意味着蒸馏训练和 CE-only 一样便宜；比较时要同时给出固定更新步数与固定训练计算预算的结果。部署时只保存学生，教师不会加入推理模型。

累积窗口先在 CPU 收集，不保留多个 GPU 前向图。CE/KL 使用 sum reduction，再以全局有效 token 数归一化。DDP 梯度默认取 rank 平均，因此每个 rank 的本地 loss sum 乘 `world_size / global_token_count`。只有窗口最后一个微批同步梯度；尾窗口使用实际有效 token 总数。

新入口只支持 dense 配置。MoE 的路由辅助损失有微批依赖，不能根据 dense 的梯度等价测试推断 MoE 整体也等价。数据 sampler 与 DataLoader 使用独立 seed，构造可选教师不会改变学生的数据顺序。

运行目录保留 `run.json`、`train.jsonl`、`model_768.pth`、`model_768.config.json`。原生推理加载 sidecar 的完整结构，并拒绝请求预算与保存预算不匹配。上游旧权重没有 sidecar，仍可用于显式的推理期压缩实验。这里支持从 checkpoint 初始化，**尚不支持精确中断续训**；不要把重新初始化 optimizer 当成 resume。

## 4. 受控实验命令

从仓库根目录运行，先按上游说明准备 `out/sft_vlm_768.pth`、固定视觉编码器与独立训练 Parquet。下面 `dataset/domain_train.parquet` 和 `dataset/holdout.jsonl` 是用户准备的数据，不是仓库自带的已标注业务数据。训练、验证、测试应按图片/文档/采集场景分组，并处理近重复。

```bash
# A：全预算继续做 CE 适配
python trainer/train_budget_vlm.py \
  --data_path dataset/domain_train.parquet \
  --init_checkpoint out/sft_vlm_768.pth \
  --image_token_len 64 --run_dir runs/ce64_s42 \
  --device cuda --dtype bfloat16 --seed 42

# B：同样样本和初始化，低预算 CE-only
python trainer/train_budget_vlm.py \
  --data_path dataset/domain_train.parquet \
  --init_checkpoint out/sft_vlm_768.pth \
  --image_token_len 16 --run_dir runs/ce16_s42 \
  --device cuda --dtype bfloat16 --seed 42

# C：B 的学生，再加冻结全预算教师监督
python trainer/train_budget_vlm.py \
  --data_path dataset/domain_train.parquet \
  --init_checkpoint out/sft_vlm_768.pth \
  --teacher_checkpoint out/sft_vlm_768.pth \
  --image_token_len 16 --kd_weight 1 --temperature 2 \
  --run_dir runs/kd16_s42 --device cuda --dtype bfloat16 --seed 42
```

CPU 使用 `--device cpu --dtype float32`。这些 CUDA 命令是可执行入口，不表示本次已经验证 CUDA AMP 性能。通过 `--max_updates` 限制开发实验；正式比较至少在独立评估集上报告多个训练 seed 的结果。小模型最先比较 16-token；4-token 可作为压力测试，不能默认越少越好。

```bash
python scripts/eval_budget_vlm.py --manifest dataset/holdout.jsonl \
  --save_dir runs/ce16_s42 --weight model --image_token_len 16 \
  --device cuda --output results/ce16_s42_actual.jsonl

python scripts/eval_budget_vlm.py --manifest dataset/holdout.jsonl \
  --save_dir runs/kd16_s42 --weight model --image_token_len 16 \
  --device cuda --output results/kd16_s42_actual.jsonl

python scripts/compare_budget_runs.py \
  --baseline results/ce16_s42_actual.jsonl \
  --candidate results/kd16_s42_actual.jsonl \
  --output results/ce_vs_kd_s42.json --bootstrap 2000 --seed 42
```

对各权重再运行 `--image_condition blank` 和 `shuffle`。按同样样本比较正确图与干预图，观察正确性变化；答案变化本身不等于模型正确使用图像。

## 5. 统计比较和失败判断

当前 evaluator 输出原问题、参考答案、原图身份、实际使用图、类别和 `group_id`。比较工具强制样本 ID 集完全相同，并逐项核对评估目标；旧版结果缺少这些字段时要求重新评测，避免静默比较两个不同集合。

Bootstrap 按图片或文档组重采样，整组保留同图问题；点估计是按样本加权的平均配对分差。默认 `group_id` 基于原图路径，跨页文档或相邻视频帧应显式填写共同场景 ID。少于 20 个独立组会警告。区间只表达该测试集的抽样不确定性，不能覆盖训练 seed 变化；单组退化区间不应被解释为高置信度。

验收优先比较 B 与 C，在相同推理预算上问“蒸馏是否改善效果”。A 用来判断低预算损失了多少能力。记录 teacher 在独立任务集上的效果，避免把错误教师当作上限；如果教师弱，应先改数据或教师，不以增加 KD 权重代替诊断。

动态 padding 缩小张量不保证按比例提速：当前注意力实现可能因 mask 切换内核路径，最长样本仍决定局部 batch 长度。训练日志中的 padded positions 是局部 rank 的位置计数，不能当 GPU FLOPs 或吞吐；elapsed_ms 包含训练测量开销，不是在线服务延迟。

没有独立任务提升时，应如实报告零结果。只有确认压缩后仍有可用证据、离线基线已稳定，才继续试 on-policy 前缀、预算课程、可靠教师位置或关键视觉特征蒸馏；一次增加一项，以免无法归因。
