# GPU 会话执行计划(数据齐备后)

前置:GPU 会话(`torch.cuda.is_available()` 为 True)、`.venv_rebuttal` 环境、
全部 `check` 已通过。以下命令均从仓库根目录执行,一次一个 GPU(-gpu 0)。

## 1. 真实显存标定(回答"多少 batch 到 70GB")

结论(代码核实):执行路径**没有 batch 参数**,全部逐行 forward(batch=1),
单进程显存 = 权重 + 单行激活,远低于 70GB。标定方式 = 跑真实 audit/fit,
从协议记录的 `peak_allocated_bytes` / `peak_reserved_bytes` 读取:

```bash
# 每个模型一次 support-only 真实审计(含 backward 门),峰值记录在 audit 记录里
.venv_rebuttal/bin/python scripts/run_aperture_cases.py audit --case E09 \
  --unit q25--camelyon17-hospital_2--n16--s0 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py audit --case E09 \
  --unit g34--camelyon17-hospital_2--n16--s0 --gpu 0 --execute
# ... 其余模型 q34 / internvl3 / phi 同理(phi 需权重下载完成)

# 汇总峰值表
.venv_rebuttal/bin/python scripts/extract_peak_memory.py
```

## 2. E03(全就绪,首选)

```bash
.venv_rebuttal/bin/python scripts/run_aperture_cases.py plan --case E03   # 已冻结,跳过
.venv_rebuttal/bin/python scripts/run_aperture_cases.py check --case E03 \
  --unit q25--hospital_2--n32--s0
.venv_rebuttal/bin/python scripts/run_aperture_cases.py audit --case E03 \
  --unit q25--hospital_2--n32--s0 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E03 \
  --unit q25--hospital_2--n32--s0 --gpu 0 --execute
# 通过后整个 E03:
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E03 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py status --case E03
.venv_rebuttal/bin/python scripts/run_aperture_cases.py summarize --case E03
```

## 3. E02(先 camelyon17 6 units,BRIGHT 数据齐后 18 units)

```bash
.venv_rebuttal/bin/python scripts/run_aperture_cases.py check --case E02 \
  --unit q25--camelyon17-hospital_2--n16--s0
.venv_rebuttal/bin/python scripts/run_aperture_cases.py audit --case E02 \
  --unit q25--camelyon17-hospital_2--n16--s0 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E02 \
  --unit q25--camelyon17-hospital_2--n16--s0 --gpu 0 --execute
# 其余 5 个 camelyon17 单元(q25/g34 × 3 seeds)同法;BRIGHT 就绪后:
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E02 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py status --case E02
.venv_rebuttal/bin/python scripts/run_aperture_cases.py summarize --case E02
```

## 4. E07(TextVQA 图片解压 + prepare 后)

```bash
.venv_rebuttal/bin/python scripts/run_aperture_cases.py prepare --case E07 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py plan --case E07
.venv_rebuttal/bin/python scripts/run_aperture_cases.py check --case E07 \
  --unit q25--textvqa--n64--s0
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E07 \
  --unit q25--textvqa--n64--s0 --gpu 0 --execute
```

## 5. 注意

- GPU 参数是 `--gpu 0`(不是 `--gpus`);执行必须加 `--execute`,否则 dry-run。
- 断点恢复 = 重跑相同命令;不要删除 plan.json / DONE.json。
- 一卡一进程(内部 GPU 锁);同一时刻只跑一个 run。
- 环境变量由 CLI 子进程自动设置(HF_HUB_OFFLINE=1、确定性 CUDA 配置)。
