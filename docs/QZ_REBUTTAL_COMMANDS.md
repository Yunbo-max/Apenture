# APERTURE Rebuttal 全部命令清单(2026-11-08)

环境:仓库根 `/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/APERTURE`,
Python 用 `.venv_rebuttal/bin/python`,batch 用环境变量 `APERTURE_BATCH`(默认 1,本次用 16)。
GPU 经 qz 平台提交(`bin/qz train CreateJob`),计算组 `lcg-df089db8-817a-4aa8-a164-eb1a32948564`(RNN 组),
spec `7166bd2e-6cbe-4bd9-be38-762d11003e7f`。

## 0. 数据链(已完成,复现用)

```bash
# BRIGHT 原始数据(zip 已下载+md5 校验+解压;ukraine/mexico/myanmar 无 pre 图已移出)
bash scripts/download_bright.sh full data/raw/bright
PYTHONPATH=src .venv_rebuttal/bin/python scripts/prepare_dataset.py bright-raster --root data/raw/bright --output data/bright.jsonl
PYTHONPATH=src .venv_rebuttal/bin/python scripts/prepare_neurips_splits.py all
PYTHONPATH=src .venv_rebuttal/bin/python scripts/prepare_visual_lens_seeds.py --output data/prepared/visual_lens/bright_fresh_20261106

# medical fresh(camelyon17 5 医院 + pathmnist;含 layout hoist 到 n16/query.jsonl)
.venv_rebuttal/bin/python scripts/prepare_medical_fresh.py configs/aperture_medical_fresh.json
.venv_rebuttal/bin/python scripts/prepare_medical_fresh.py configs/aperture_medical_fresh_h0134.json --target-root runs/aperture_medical_fresh_20261106/prepared
.venv_rebuttal/bin/python scripts/prepare_pathmnist_fresh.py

# comprehensive prepared(派生配置含 20/22 数据集)
PYTHONPATH=src .venv_rebuttal/bin/python scripts/run_aperture_comprehensive.py --config configs/aperture_comprehensive_prep_e02.json prepare

# E03 支持集 / E07 TextVQA / manipbench / guardian
.venv_rebuttal/bin/python scripts/run_aperture_cases.py prepare --case E03 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py prepare --case E07 --execute
PYTHONPATH=src .venv_rebuttal/bin/python scripts/prepare_manipbench_generalization.py --seed {0,1,2}
PYTHONPATH=src .venv_rebuttal/bin/python scripts/prepare_guardian_failure.py --data-root data/guardian_ood

# 冻结计划(run_root = runs/aperture_rebuttal_cases_20261108_b16)
for c in E01 E02 E03 E04 E05 E06 E07 E08 E09; do
  .venv_rebuttal/bin/python scripts/run_aperture_cases.py plan --case $c
done

# check 全部(数据/模型/依赖前置,无 GPU)
.venv_rebuttal/bin/python scripts/run_aperture_cases.py check --case E02   # 18 units
.venv_rebuttal/bin/python scripts/run_aperture_cases.py check --case E03   # 36 units
.venv_rebuttal/bin/python scripts/run_aperture_cases.py check --case E04   # 9
.venv_rebuttal/bin/python scripts/run_aperture_cases.py check --case E05   # 18
.venv_rebuttal/bin/python scripts/run_aperture_cases.py check --case E06   # 2
.venv_rebuttal/bin/python scripts/run_aperture_cases.py check --case E07   # 6
.venv_rebuttal/bin/python scripts/run_aperture_cases.py check --case E09   # 15
# E01 按 domain 逐个 check(已就绪的 20 个域)
.venv_rebuttal/bin/python scripts/run_aperture_cases.py check --case E01 --domain <domain>
```

## 1. 执行(全部 GPU 阶段,一单元或整 case)

```bash
export APERTURE_BATCH=16

# E03 全量(36 单元;双卡:--model q25 与 --model g34 各一个作业)
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E03 --model q25 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E03 --model g34 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py status --case E03
.venv_rebuttal/bin/python scripts/run_aperture_cases.py summarize --case E03

# E02 全量(18 单元;双卡:--model q25 与 --model g34)
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E02 --model q25 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E02 --model g34 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py status --case E02
.venv_rebuttal/bin/python scripts/run_aperture_cases.py summarize --case E02

# E04(9 单元,q25) / E05(18 单元,双卡按模型) / E06(2 单元,q25) / E07(6 单元,双卡按模型)
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E04 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E05 --model q25 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E05 --model g34 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E06 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E07 --model q25 --gpu 0 --execute
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E07 --model g34 --gpu 0 --execute

# E09 五模型支持集审计(15 单元,单卡顺序)
.venv_rebuttal/bin/python scripts/run_aperture_cases.py audit --case E09 --gpu 0 --execute

# E01 已就绪部分(20 域 × 5 模型 × 3 seeds = 300 单元;按 domain 分作业)
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E01 --domain <domain> --gpu 0 --execute

# E08 分析(E01 四个 BRIGHT 证据齐后)
.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E08 --execute
```

## 2. qz 提交模板(单卡 H100 作业)

payload 参考 `scripts/qz_e03_full_payload.json`(LCG/spec 已是 RNN 组),
命令核心:

```json
{
  "command": "set -euo pipefail\ncd /inspire/hdd/global_user/zhangjiaquan-253108540222/backup/APERTURE\nexport APERTURE_BATCH=16\n.venv_rebuttal/bin/python scripts/run_aperture_cases.py run --case E03 --model q25 --gpu 0 --execute 2>&1 | tee runs/qz_jobs/<name>.log\necho JOB_DONE",
  "max_running_time_ms": "172800000",
  "framework_config": [{"image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
     "image_type": "SOURCE_PRIVATE", "instance_count": 1,
     "spec_id": "7166bd2e-6cbe-4bd9-be38-762d11003e7f", "shm_gi": 100}]
}
```

提交:`/inspire/hdd/global_user/zhangjiaquan-253108540222/bin/qz train CreateJob --data @scripts/<payload>.json`
查状态:`... qz train ListJobs --workspace-id ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6 --page-size 10 --page-number 1`

## 3. 断点恢复与注意

- 重跑相同命令 = 校验后复用已完成结果(不要删 plan.json / DONE.json)
- `--gpu 0`(不是 --gpus);执行必须加 `--execute`
- 一卡一进程(内部 .gpu-N.lock)
- E02 需要官方 pyreft(已装 no-deps 版);DoRA 需要 peft use_dora(已装)
- E03 的 summarize 在未完成全部 36 单元时会拒绝完整汇总(预期保护)
- 源码改动会作废冻结计划与已封存结果(须重新 plan+check+run)
