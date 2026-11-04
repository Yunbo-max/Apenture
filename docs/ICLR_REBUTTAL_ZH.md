# APERTURE rebuttal：正确仓库与补实验入口

目标是 `Yunbo-max/Apenture`，不是 SPECTRUM。本次只增加实验编排、回归测试和仓库标题/作者元数据；不修改 APERTURE 算法、已有结果、历史配置或论文正文。SPECTRUM 的 532 项 CPU 测试和其运行命令不适用于本仓库。

## 固定范围

复用 `docs/EXPERIMENT_MATRIX.md` 已经约定的两项补实验，不另立方法：

| suite | 范围 |
|---|---|
| `medical-multihospital` | CAMELYON17 五医院 0–4，support seeds 10/11/12，24 fit + 8 calibration，500 queries；Qwen2.5-VL-7B 与 InternVL3-8B，135 个预定状态、105 次可训练拟合 |
| `efficiency-rank16` | Hawaii/H2、seed 0，Frozen/LoRA-1E/LoRA-4E/APERTURE；复用 rank-16 final worker 的两次 warm-up、12 个完整 candidate-scoring queries、3 次计时重复 |

医疗默认保留 `main` 和 `replication` 两阶段，不自动运行可选 16/64-label curves。效率组保留原 final worker 的评分、校准和诊断，并非新造一个省略步骤的微基准。它需要原 Hawaii/H2 support/query manifests 已存在。

已完成的 BRIGHT、PathMNIST、robot 等记录不自动重跑。参考 `docs/RUNBOOK.md` 和 `docs/CLAIM_FREEZE.md`。负结果仍保留，query 标签不能用于拟合或挑参数。

## 本地操作

在独立 checkout/环境中使用固定提交，不修改正在运行的旧 checkout。先准备满足原协议的 GPU 环境与模型、数据路径；缺数据、权限或显存时停止，不换小模型或降分辨率。

```bash
python -m pip install -e '.[train,test]'
bash scripts/check_local.sh
python -m pytest -q tests/test_rebuttal_entrypoint.py
```

在第一次生成计划前，按已有文档配置这两个源文件中的本机路径：

- `configs/aperture_medical_multimodel.json`
- `configs/aperture_final_round.json`

保留医院、seed、标签/查询预算和模型方案。然后：

```bash
# 只生成封存配置和显示命令；不会加载模型或启动实验。
python scripts/run_aperture_rebuttal.py plan \
  --root runs/aperture_rebuttal_20261104 --gpus 0

# 前台执行两项补实验；每项顺序 prepare -> plan -> check -> run -> summarize。
python scripts/run_aperture_rebuttal.py run \
  --root runs/aperture_rebuttal_20261104 --gpus 0 --execute

# 只调用原始汇总器，不补跑训练。
python scripts/run_aperture_rebuttal.py summarize \
  --root runs/aperture_rebuttal_20261104 --gpus 0 --execute
```

仅做医疗或效率组时，从首次 plan 起使用对应 `--suite medical-multihospital` 或 `--suite efficiency-rank16`，并使用独立 root；后续保持同样参数。多 GPU 使用 `--gpus 0,1`，从首次 plan 开始固定。命令不带 `--execute` 一律不启动子进程；即使 plan 带上该开关也不会运行实验。

新入口只修改派生配置里的 `run_root` 和 `prepared_root`；效率组另外固定原协议声明的 Hawaii/H2、seed 0 子集，不改变原始 manifests。派生配置存于新 root 的 `configs/`，未写入历史配置；原文件修改会触发冻结检查。

## 失败与恢复

代码、配置、Python 路径、GPU 列表和 suite 列表写入不可变 `plan.json`。程序在每个阶段前检查代码与配置身份，已改变就停止。不要删 manifest 强行兼容；使用原 checkout 或新 root。

重跑完全相同的 run 命令会重新调用原 suite 的完整检查顺序，是否复用结果由原 suite 的 seal/receipt 检查决定。本编排器不会仅凭 exit=0 跳过旧拟合，也不重新实现 checkpoint 恢复。

同一个 root 使用独占锁；确认旧进程确已退出才可手动清理残留锁。失败立即停止后续阶段，并在 `attempts/` 保存动作与退出码。成功状态 `commands_completed` 仅表示子命令正常返回，不自动证明结果齐全或论文结论成立；原始汇总器的结果才是科学证据。

## 验证边界

此次在隔离环境中执行新增入口的 15 项离线回归测试，全部通过；覆盖配置保持、历史目录保护、代码/config 修改拒绝、dry-run、失败停止、锁定 GPU 参数和仅汇总路径。没有下载真实模型、运行 GPU、生成新的 benchmark 分数，且没有在此环境重新跑 Apenture 的全部历史测试。保留本机 `scripts/check_local.sh` 与真实 GPU audits；不新增 GitHub Actions。

README/CITATION 已使用用户指定的标题和九位作者；这不是把完整作者版 ZIP/PDF 上传到仓库。`paper/visual_lens/` 仍是历史源快照，不能用这里的元数据更新宣称 manuscript 已重新编译。当前快照的 `main.tex` 引用 `sections/part_01.tex` 等文件，但对应 sections 目录未出现在本次读取的 Git tree 中；论文构建应使用已单独交付的完整作者版包，不要将旧快照误当作该包。
