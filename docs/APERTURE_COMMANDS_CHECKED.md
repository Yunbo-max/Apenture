# APERTURE 命令复核（2026-11-07）

## 这次确实修复了什么

`490ee72` 的 case 入口参数名与文档一致，但 LoRA/DoRA 包装器存在真实接口错误：原生 `loaded()` 返回 `(model, backend, controller)`，新增包装器却把第三项当作 processor。Frozen 路径第三项是 `None`，导致 LoRA/DoRA 新建 Backend 时拿不到 processor。本次改为读取 `backend.processor`，覆盖训练与重载两条路径。APERTURE 算法、损失、基提取和实验范围没有改变。

默认 `run_root` 改为 `runs/aperture_rebuttal_cases_20261107_v2`，避免修复后的源码碰到 v1 的封存计划；`prepared_root` 仍复用 v1 数据。不要删除旧 plan/DONE 记录来强行恢复。已启动的旧实验保留旧 checkout；新实验使用修复提交与新 root。

## 验证范围

- 原有新增用例测试：28 passed。
- 新增 native-loader 合同测试先得到4个预期失败（LoRA/DoRA，训练/重载）；修复后通过。
- 两个相关测试文件合计48 passed、0 failed、0 skipped。
- 真实 CLI 子进程检查47次：帮助、列表、全部 E01–E09 计划、dry-run、状态、单单元选择和预期拒绝均符合预期。
- CLI 测试使用 GitHub 原始 comprehensive 配置（Git blob `997e2489f38e73c54746ef4fb796df9ff0545004`），不是缩小过的临时实验配置。
- 没有真实模型权重/私有数据，没有安装完整 native VLM 依赖栈，也没有 GPU、真实 `check` 成功、训练或 benchmark 成绩。LoRA/DoRA 的加载接口回归使用忠实于原接口的测试替身；不等同于真实 PEFT 集成验收。没有重新运行仓库全部历史测试。

## 环境与路径必须先检查

从完整 Apenture checkout 的根目录，在已配置好的 Linux/WSL 训练环境中执行。不能只解压 additions ZIP 后直接训练。新 checkout 不会自动带上服务器上的模型和未追踪数据。

**不要盲目重跑旧的 `pip install -e '.[train,test]'` 命令来覆盖环境。**仓库现有 `[train]` 仍限定 `peft>=0.14,<0.15`，而 comprehensive 的 readiness 文档记录 PEFT 0.17.1。这个依赖范围没有在本补丁中改写；本次也没有验证一个全新的 CUDA 安装组合。保留已审计环境，在独立环境中补齐缺失依赖再做真实审计。

E02 需要官方 `pyreft.LoreftIntervention`，DoRA 需要 PEFT 的 `use_dora` 选项。可先做导入检查（它不是模型/GPU审计）：

```bash
python - <<'PY'
import inspect
import torch, transformers, peft
from packaging.version import Version
from peft import LoraConfig
from pyreft import LoreftIntervention
assert Version('4.57.1') <= Version(transformers.__version__) < Version('5')
assert 'use_dora' in inspect.signature(LoraConfig).parameters
print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
print('transformers:', transformers.__version__, 'peft:', peft.__version__)
print('LoReFT import:', LoreftIntervention.__name__)
PY
```

在第一次 `plan`/`run` 前核对两个配置：

- `configs/aperture_comprehensive_v1.json`：`models[*].path` 与旧 comprehensive 的 `prepared_root`；这些才是 E01/E02/E04/E05/E06/E09 使用的模型和 prepared manifests 路径。
- `configs/aperture_reviewer_cases_v1.json`：`base_config`、新输出 root、E03 官方 CAMELYON17 原始数据路径、E07 原始 TextVQA JSON/图片路径。

E01/E02/E04/E05/E06/E09 不负责下载/自动补齐 comprehensive 数据；缺失时停止。E03/E07 的 `prepare` 也不下载模型和原始数据。不要用更小数据量来让 check 通过。

## 先跑一个 E02 单元

```bash
python scripts/run_aperture_cases.py --help
python scripts/run_aperture_cases.py list
python -m pytest -q tests/test_reviewer_cases.py tests/test_reviewer_command_checks.py

python scripts/run_aperture_cases.py plan --case E02

# 无 --execute：只列出匹配单元；不验证本机模型/数据，不启动GPU。
python scripts/run_aperture_cases.py run --case E02 \
  --unit q25--camelyon17-hospital_2--n16--s0 --gpu 0

# 真实模型/数据/依赖前置检查；没有模型 forward。
python scripts/run_aperture_cases.py check --case E02 \
  --unit q25--camelyon17-hospital_2--n16--s0

# 支持集上的真实GPU审计，不训练、不评分query。
python scripts/run_aperture_cases.py audit --case E02 \
  --unit q25--camelyon17-hospital_2--n16--s0 --gpu 0 --execute

# 该完整比较单元：审计、全部预定候选拟合、校准选择、恢复与query评测。
python scripts/run_aperture_cases.py run --case E02 \
  --unit q25--camelyon17-hospital_2--n16--s0 --gpu 0 --execute
```

上条回复的等价选择也正确：`--model q25 --domain camelyon17-hospital_2 --seed 0`。单单元包含完整基线/候选比较，并不是仅跑一个模型 forward。

通过代表单元后，单独启动整个 E02：

```bash
python scripts/run_aperture_cases.py run --case E02 --gpu 0 --execute
python scripts/run_aperture_cases.py status --case E02
python scripts/run_aperture_cases.py summarize --case E02
```

`status` 当前主要显示拟合覆盖与 selection seal，不是完整 query 验收。只有整个 case 的 `summarize` 成功才表明完整结果通过汇总校验。未完成时返回非零并写 coverage 是预期行为；不用删数据重来。普通 case 的 summarize 不需要 `--execute`。

## E03 与 E07 的单独准备

```bash
python scripts/run_aperture_cases.py prepare --case E03 --execute
python scripts/run_aperture_cases.py plan --case E03
python scripts/run_aperture_cases.py check --case E03 --unit q25--hospital_2--n32--s0
python scripts/run_aperture_cases.py run --case E03 \
  --unit q25--hospital_2--n32--s0 --gpu 0 --execute

python scripts/run_aperture_cases.py prepare --case E07 --execute
python scripts/run_aperture_cases.py plan --case E07
python scripts/run_aperture_cases.py check --case E07 --unit q25--textvqa--n64--s0
python scripts/run_aperture_cases.py run --case E07 \
  --unit q25--textvqa--n64--s0 --gpu 0 --execute
```

注意：E03 的 domain 是 `hospital_2`，不是其他分类 case 使用的 `camelyon17-hospital_2`。只用 E03 的 model/domain/seed 会选中16、32、64三个预算；用上面的完整 unit 才能只跑32标签。E07 固定64标签。两项 prepare 都是各自整套预定数据准备，不接受执行过滤来缩小研究范围。

## 其余 case 独立命令

以下每行独立选择执行，不是建议全复制成串行脚本：

```bash
python scripts/run_aperture_cases.py run --case E01 --gpu 0 --execute
python scripts/run_aperture_cases.py run --case E04 --gpu 0 --execute
python scripts/run_aperture_cases.py run --case E05 --gpu 0 --execute
python scripts/run_aperture_cases.py run --case E06 --gpu 0 --execute
python scripts/run_aperture_cases.py audit --case E09 --gpu 0 --execute

# 仅在 E01 预定的 Qwen/四BRIGHT/三seed/五方法证据齐全后。
python scripts/run_aperture_cases.py run --case E08 --execute
```

E08 是只读分析，不用GPU，也不重新训练；没有 `--execute` 只返回 dry-run。E09 只做 support audit，没有拟合/query状态。

每个GPU进程使用单数 `--gpu 0` 或 `--gpu 1`，**不是** `--gpus`，也不接受 `--gpu 0,1`。多GPU用独立命令选择不重叠的 unit/case。`run` 自带校验后恢复，没有 `--resume` 参数；`--root` 也不是该入口参数，输出目录在配置中指定。相同 root 内有 GPU/unit 锁，不能阻止其他程序或另一 root 占用同一GPU。

最终结论：命令格式和选择范围已实际检查，确认的 processor 接口错误已修复；真实环境的模型/数据 check 与代表单元GPU审计仍必须在实验服务器上执行，不能从 CPU/dry-run 推出GPU实验已成功。
