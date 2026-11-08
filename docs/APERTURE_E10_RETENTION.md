# E10 — 少样本适配后的原有能力保留

这是一套**只读评测扩展**，不是新的 APERTURE 算法。它不训练、不改损失、不加梯度手术，不自动得出“不遗忘”或“优于 LoRA”的结论。先完成/封存原来的 E01/E02 拟合与 E02 校准选择，再复用对应状态。原始 APERTURE 固定 rank16、layers [14,27]、alpha3、lr0.05、4次更新；不得把调参后的变体重新标为原版。

为避免破坏正在运行的 E01–E09 source fingerprints，新增实现放在 `scripts/aperture_retention/`，入口为 `scripts/run_aperture_retention.py`。**不修改任何现有 src 文件、旧计划、既有入口、数据、适配器或 Actions。**现有全 src 指纹与模型/环境/manifest/fit seal 必须匹配；新 evaluator 单独封存。来源版本混杂时停止，不删除旧 plan 强行继续。

## 三部分，分别启动与保存

| part | 固定范围 | 解释 |
|---|---|---|
| `visual`（E10-A） | Qwen2.5-VL-7B、Gemma3-4B；H2/Hawaii/Libya 独立适配；seeds 0/1/2，共18单元 | 新域模块一直开启，同时测新域收益和其他视觉域退化 |
| `text`（E10-B） | 同18个适配单元；完整 ARC-Challenge test 1172、BoolQ validation 3270 | 模块保持加载；APERTURE 每个文本 forward 使用全零视觉 mask；LoRA 同时给 active 和 disabled 两种结果 |
| `bank`（E10-C） | 两模型、三个seed；Hawaii→Libya→Noto→Turkey，4个保存模块数量，共24单元 | **独立训练的归档模块恢复模拟**：两种方法都知道域ID，加载各自旧模块；不冒充单一共享适配器的持续学习 |

A/B 每个单元包含 Frozen、原版 APERTURE、LoRA-r16 1/4 passes、E02 calibration-only 选择的 LoRA-r4。r4 必须使用与 E01 相同的 fit/calibration/query 和模型文件。E02 选择 receipt 与所有 r4 候选重新验证；不得基于 E10 结果挑选。

A 的目标：H2→H0/H1/H3/H4+PathMNIST；Hawaii→Libya/Noto/Turkey；Libya→Hawaii/Noto/Turkey。每次也评测源新域 query，防止把“未学到新域”误当作抗遗忘。跨域时保留同一已训练状态、mask、层和输入处理规则；不做逐query校准、不使用其他域温度。query 与源 fit/calibration 的样本、组及像素哈希不允许重叠。所有新指标为 raw candidate softmax。

B 的 ARC 候选是答案文本，BoolQ 候选是 no/yes；统一使用 native text chat template，无few-shot例子、图像、OCR或标签提示。评分为答案 token log-probability 之和，不做长度归一化、候选挑选或截断。不是新的官方 leaderboard 分数声明。无视觉输入时 m=0 的等价性必须通过实测 logits/预测检查，不仅靠 frozen 参数断言。LoRA-disabled 恢复能力必须报告，不能被当成 LoRA-active 的成绩。

C 仅比较 APERTURE、LoRA-1pass、LoRA-4passes（r4 没有完整四域源计划，不伪造）。每个数量阶段在 fresh process 中重新加载可用的每一个旧模块，对完整旧域 query 与原始 E01 封存预测做配对恢复检查，统计包含固定基 U、原始控制矩阵、deployment metadata 和 LoRA adapter 文件的累计字节数，共享基座双方都不重复计。它**不是在E10中连续训练四次**；若双方都恢复准确，结论是模块化保留/存储折中，而不是 LoRA 遗忘、APERTURE 不遗忘。

## 来源配置先改好

`configs/aperture_retention_v1.json` 的默认来源为 `runs/aperture_rebuttal_cases_20261108_b16/{E01,E02}/plan.json`。若实际结果分散在 per-card root，把那些真实的 E01/E02 plan 路径全部加入 `source_plans`。同一个比较单元必须恰好有一个完整来源；不按成绩挑选、不把多个不完整来源拼成一个单元。重复完整来源会拒绝，应在配置中显式消除歧义。

`base_config` 的 prepared_root 指向原综合数据。不得改变旧源配置来迁就E10；新输出固定在独立 `runs/aperture_retention_20261108_v1/`。有未完成的E01/E02时继续用原入口拟合，E10不会自动训练。A/B 所需源域都是 E02 的原定 panel。C 还需要 E01 的 Noto/Turkey **完整 evaluation receipt**，否则停止。

## 命令

在完整仓库根目录、与原拟合一致的Linux/WSL环境中执行；不要升级/降级正在训练的旧环境。新增文本数据准备需要 `datasets`，不会自动安装依赖。

```bash
python scripts/run_aperture_retention.py list
python -m pytest -q tests/test_retention_e10.py

# A：plan 封存完整18单元，过滤器只选择执行，不缩小报告母体。
python scripts/run_aperture_retention.py plan --part visual
python scripts/run_aperture_retention.py check --part visual \
  --unit q25--camelyon17-hospital_2--n16--s0
# 无 --execute 只列出将执行的单元，不加载模型。
python scripts/run_aperture_retention.py run --part visual \
  --unit q25--camelyon17-hospital_2--n16--s0 --gpu 0
python scripts/run_aperture_retention.py run --part visual \
  --unit q25--camelyon17-hospital_2--n16--s0 --gpu 0 --execute
# 完整A；通过代表单元后再运行。
python scripts/run_aperture_retention.py run --part visual --gpu 0 --execute
python scripts/run_aperture_retention.py status --part visual
python scripts/run_aperture_retention.py summarize --part visual

# B：显式下载固定revision的完整文本集，然后计划和运行。
python scripts/run_aperture_retention.py prepare-text --execute
python scripts/run_aperture_retention.py plan --part text
python scripts/run_aperture_retention.py run --part text \
  --unit q25--camelyon17-hospital_2--n16--s0 --gpu 0 --execute
python scripts/run_aperture_retention.py run --part text --gpu 0 --execute
python scripts/run_aperture_retention.py summarize --part text

# C：模块库恢复模拟，不训练；例子是一个seed的第四阶段。
python scripts/run_aperture_retention.py plan --part bank
python scripts/run_aperture_retention.py run --part bank \
  --unit q25--bank--s0--t4 --gpu 0 --execute
python scripts/run_aperture_retention.py run --part bank --gpu 0 --execute
python scripts/run_aperture_retention.py summarize --part bank
```

所有命令可加 `--config path/to/local.json`。`--gpu` 是单数，只允许一个序号；多卡用不重叠单元的独立命令。OS GPU锁仅覆盖该E10 root，不能阻止别的程序占用同一卡。无后台任务。`--resume` 和 `--root` 不是参数；相同命令校验后恢复逐预测完整行，坏尾行显式报错，不丢弃。准备文本数据会下载，run强制HF离线。摘要要求整个part齐全，不能用模型/seed过滤器选择性汇总。

## 运行与证据合同

`check` 会核对源代码、模型、环境、fit/selection/audit receipts、数据划分与像素，不做模型forward。每次真实运行先用原来的calibration预测检查恢复后的分数（max abs <= 0.005，失败不得放宽阈值）。恢复检查使用源fit记录的batch_size；新增query统一逐行评分，避免变长候选batch行为不明确。模型始终BF16，不自动quantize/offload/降分辨率。

每个单元保存 `binding.json`、`predictions/<task>/<method>.jsonl`、`result.json`、`DONE.json`，part另有 `logs/` 和 `report/`。真实标签只进入评分汇总，不传入原生候选forward；适配query标签和保留任务标签从不参与选择。源文件eval前后重验，结果不回写E01/E02。

报告：新域accuracy/macro-F1增益、每个旧域accuracy/macro-F1下降、raw NLL/Brier、原来正确→错误比例、原来错误→正确数量、预测变化、文本identity检查、完整状态字节数；query-group paired bootstrap和三个support seed均值/SD分开。Frozen全错时“丢失原正确比例”为null，不假装0。小患者组数明确警告。不能通过“不显著”宣称无遗忘；没有预注册非劣检验。

文本/模块恢复 identity 失败作为结果保留并明确标记；不能隐藏这些单元。源calibration重载失败则停止，这属于模型/实现合同不匹配。完整汇总在读取之前撤销旧aggregate，任何缺失/损坏都阻止完整科学汇总，状态文件如实保留。

## 数据依据与评审关联

文本集使用作者发布的固定版本：
- https://huggingface.co/datasets/allenai/ai2_arc/tree/d67d66ae44ae69ff0c82ba64779a3736b2310ce4 (ARC-Challenge/test,1172)
- https://huggingface.co/datasets/google/boolq/tree/9aa7c5589ad6dbb19e78b619ed88a79f06e99aad (validation,3270)

E10是新增的性能/干扰/存储证据，可补R1的收益定位和域间差异、R2的实际适配价值；三位评审原文没有明确要求持续学习实验，不应声称E10替代强基线、support曲线或方法澄清。

## 当前验证边界

离线测试与CLI检查记录见随交付的验证包。没有原生7B/4B权重、私有数据或CUDA运行；不声称GPU集成通过，不声称APERTURE少遗忘或优于LoRA。不得把E10设计写成已完成结果。先在服务器跑代表单元check/run，再展开其余单元。
