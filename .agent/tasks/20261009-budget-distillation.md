# V2 investigation record

User request: improve MiniMind-V further using current academic and production practice.
Starting branch: feat/budgeted-vision-lab, V1 commit 4b4201177cb8e7bb2facc44ef1891a7639ab7aa4.

Observed evidence: fixed-padding baseline does not shorten decoder training tensors;
unequal microbatch token counts give a 0.25 gradient error in a minimal two-class example.
After downloading pinned public weights, one demo image changes from a dog-related
generation at 64/16 tokens to an aircraft-related prefix at 4 tokens. This is a case study,
not an accuracy estimate. Original six prefixes are preserved in docs/evidence.

Intervention: train fixed low-budget student with same-vocabulary full-budget teacher,
same retained text, aligned answer-token KL and CE. Pre-projector frozen ViT features are
shared once per microbatch. This tests adaptation to reduced visual evidence; it is not a
new token selector and does not implement on-policy rollouts or 3D supervision.

Validation includes whole-batch versus unequal-microbatch gradients, two-process CPU
Gloo normalization, full tiny real-SigLIP-module CLI training, config roundtrip, strict paired
bootstrap joins, and a single full-size pretrained optimizer update. See VALIDATION_V2.

Remaining scientific work: true independent labels and group splits, quality/cost Pareto
measurements, multiple training seeds, trained-teacher qualification and explicit ablations.
Engineering unknowns: real CUDA/NCCL VLM training, AMP behavior, exact resume and
deployment engine integration. Current CPU tests are not substitutes for these checks.
