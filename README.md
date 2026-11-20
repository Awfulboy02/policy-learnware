# Policy Learnware v0

用上传策略在来源任务上的执行轨迹建立 RKME。复用者在本地把有限的目标专家轨迹压缩为需求规约，平台用核距离召回冻结策略，然后复用者用同一批日志执行 FQE 或 AM，通过估计的目标任务回报选出一个策略。代码在一台机器上模拟各方的数据流。独立的评分命令读取留出的真实闭环回报。

## 运行

Python 3.10+、NumPy 和 PyTorch。在仓库根目录可用 `python -m pip install -e .` 安装命令入口。已有依赖的环境也可用 `PYTHONPATH=src python -m policy_learnware.cli` 调用相同命令。

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

以下是可重跑的 RingTrack 小例子。它枚举两个区域各两个初始状态，精确计算两种冻结策略的有限时域回报。示例中的单步拟合只检查数据流。

```sh
policy-learnware prepare ringtrack --out runs/ring
policy-learnware specify --source runs/ring/source.json --settings runs/ring/settings.json \
  --out runs/ring-catalog --size 5 --iterations 1
policy-learnware rank --catalog runs/ring-catalog/catalog.json --requests runs/ring/requests.json \
  --request left --method fqe --k 2 --steps 1 --out runs/fqe.json
policy-learnware score --ranking runs/fqe.json --truth runs/ring/truth.json --out runs/fqe-score.json
```

`specify` 将一组不可混用的规约一次写入新目录。换不同的核、支撑点数或随机种子时请使用另一个输出目录。

## Walker 数据

```sh
policy-learnware prepare walker \
  --archive /share/songyf/RL_Learnware/experiments/b_v2_data_20260909 \
  --out /share/songyf/policy-learnware-runs/v0/walker
```

清单引用原始文件的绝对路径，冻结权重从旧项目对应的 `assets/b_v2_20260909` 读取。排序只加载召回候选的权重，策略接收原始观测并使用自身检查点中的标准化参数。

| 清单 | 角色与切分 |
|---|---|
| `source.json` | 每个候选用编号 0–5 的来源轨迹做回报准入，6–11 专用于规约。准入要求平均回报至少 900，且独立归档来源专家数据上的行为 MSE 至多 0.01。文件保留所有候选的准入结果。 |
| `requests.json` | 20 个初始分布条件按原始组保留 12/4/4 开发、验证和历史测试划分。每个条件 8 条 SAC 专家轨迹。按整条轨迹取预算，各种方法共用同一随机种子的同一子集。 |
| `truth.json` | 每个条件和每个合格候选各有 4 条配对效果轨迹，另有 4 条参考专家轨迹。只有 `score` 读取这些回报，排名落盘后才评分。 |

每份 NPZ 是一个从初始状态开始的有序 episode。常规字段为 `observations, actions, rewards, next_observations, terminated, truncated`。完整的有序 `(s,a,r)` 也可按相邻行重建后继状态。缺少最后一个后继时，需能判定已到评估时域或自然终止。AM 会跳过这个缺失的动力学标签。提前中断的日志不足以描述余下的占用，RKME 规约要求完整评估时域或真正终止。

## 方法与对照

- **规约**采用 Gaussian 核的 reduced KME。用加权 k-means 初始化支撑点，交替求解允许为负的核权重并更新支撑点。每个访问点权重为 `gamma^t / (轨迹数 × sum_t gamma^t)`，真正终止后的占用为零。上传和需求共用公开的特征变换、核带宽和时域。Walker 默认使用原始 24 维观测与 6 维动作，RBF 带宽为 sqrt(30)。这些是起始设置，可在开发条件上确定后冻结。完整需求经验 KME 对照用 `--empirical` 在本地计算。
- **FQE**为每个候选分别拟合带剩余时域信息的 Q，Bellman 目标使用该候选在下一状态的动作。取用户日志初始状态的平均估值。自然终止或到达 H 时停止自举。Walker 评分使用 H=1000、gamma=1。
- **AM**为每个请求及数据预算拟合一次状态增量和奖励模型。存在自然终止样本时同时拟合终止概率。所有候选共用模型，从相同初始状态在学习模型中推演。本版为确定性均值动力学近似。
- **对照**可使用 `--method rkme` 直接取最近候选。对于 `--method fqe` 或 `am`，`--retrieval rkme|random|all` 分别使用核召回、随机等量候选和全池。`--budget` 控制完整目标专家轨迹数，`--k` 控制召回数，`all` 总是评估整个合格池。
- **结果**记录入围和选中策略、距离、估值、数据预算对应的轨迹索引与耗时。共享 AM 拟合与候选评分分开计时。total 是读取清单后本地处理的墙钟时间，含当前进程首次加载估值模块的开销。它不测网络传输。完整经验 KME 对照计算查询的核自相似项，比较检索效率时应单独说明这项开销。评分报告逐条闭环回报，并把总 regret 分成召回损失（全池最好减去入围最好）和重排损失（入围最好减去最终选中）。估值或距离完全相同时按候选 ID 稳定排序。

来自不同初始条件的来源和需求 RKME 距离描述观测到的行为占用。候选在目标上的闭环表现要用留出的真实回报验证。少量专家数据对候选动作覆盖不足时，FQE 和 AM 均可能外推失准。Walker 的 20 个条件共享同一原生任务，历史测试适合做回顾性评估。
