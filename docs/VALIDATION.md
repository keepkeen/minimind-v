# 验证记录

> 本文件保留 V1 历史状态。V2 已补充训练、配置和真实权重探针，最新记录见 [VALIDATION_V2.md](VALIDATION_V2.md)。

日期：2026-10-09。基线：`1862b633fc082a723e78dbead9777545f09d960c`。

## 已运行

在 macOS ARM64、Python 3.12.13、PyTorch 2.6.0、transformers 4.57.6、datasets 3.6.0、pyarrow 23.0.0 的隔离环境中运行 CPU 测试：

```text
python -m pytest -q tests
23 passed in 66.34s
```

这段时间是整个测试进程耗时，包含首次依赖初始化，不是 VLM 的吞吐测试。

另外已执行 `git diff --check` 和 Python `compileall`，均通过；原始推理、新 QA 评测、pretrain 与 SFT 四个入口的 `--help` 都以退出码 0 返回，并包含 `--image_token_len` 参数。这不等同于跑过完整训练或真实 checkpoint 推理。

覆盖内容包括 64 档 projector 等价与权重 key、4/16/64 输出形状、空间池化布局、梯度传递、相邻多图融合、占位符不匹配拒绝、变长多图/纯文本 mask、raw/dict/legacy 图像张量、视觉特征缓存、KV cache 一致性、冻结视觉编码器状态、多返回序列顺序、真实 tokenizer 与 Parquet、padding 标签、答案截断、EM/ANLS-style，以及 actual/blank/shuffle 三种评测入口的合成模型冒烟测试。

测试用的是小型随机语言模型、合成视觉编码器和临时人工图片；真实 tokenizer 与真实 Parquet 文件用于数据链路验证。没有加载上游完整预训练权重，所以这些测试不证明识图、OCR 或问答效果。

## 上游 bug 的独立最小复现

从上述基线 commit 取出原始 `count_vision_proj` 运行：一个 132 长度输入，包括两个连续图像的 128 个 marker，加 4 个文本位置。

```text
input_sequence_length: 132
upstream_output_length: 68
patched_output_length: 132
```

新的融合函数不再将相邻图像合并为一个 marker run；数量校验不允许静默缩短序列。

## 默认 dense 参数计数

直接构造默认 VLMConfig，使用 meta device 统计非视觉参数，不下载权重：

```text
LLM:                 63,912,192
projector:            1,182,720
nonvision total:     65,094,912
freeze_llm=1 total:  15,931,776
```

完整视觉编码器参数量未在本次测试中独立计数；手册中的约 95M / 整机约 160M 来源于上游 README，明确为该口径。

## 发现的环境问题

第一次使用本机 Python 3.14.7、torch 2.12.0 和上游 datasets 3.6.0 时，视觉测试通过，但三个 Parquet 测试在依赖 dill 的序列化路径报错。没有修改测试绕过 Parquet，而是创建 Python 3.12 的独立环境，使用上游 PyTorch 2.6.0 参考版本重新运行全部测试。

两个虚拟环境都被 gitignore 排除；没有更改全局 Python 依赖。

## 未验证和未完成

没有执行完整 pretrain/SFT、真实 checkpoint QA 基准、GPU 显存/延迟测量、CUDA AMP、多卡 DDP、torch.compile、移动端量化、导出模型的跨版本兼容矩阵或生产服务压测。没有训练获得新的效果权重，也没有宣称 16/4 token 准确率、无损压缩或加速百分比。

GitHub Actions 的 CPU 测试工作流已加入。除非实际运行结果另有记录，本文件不将远程 CI 状态与本地测试状态混为一谈。

已知后续工程工作还包括大数据吞吐、多种子入口、正式数据集 evaluator、原生权重的完整配置 sidecar，以及更严格的断点恢复。原训练循环的梯度累积尾部处理和随机状态恢复也需要独立审计。本次测试不等于完整训练系统认证。
