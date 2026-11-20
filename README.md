# Policy Learnware v0.1

用上传策略在来源任务上的执行轨迹建立 RKME。复用者在本地把有限的目标专家轨迹压缩为需求规约，平台用核距离召回冻结策略，然后复用者用同一批日志执行 FQE 或 AM，通过估计的目标任务回报选出一个策略。代码在一台机器上模拟各方的数据流。独立的评分命令读取留出的真实闭环回报。

## v0.1 扩量资产与权限面板

三个原生任务分别检索。用 `prepare-am10` 将冻结源资产、需求轨迹和外部评分索引成三套清单。在共享服务器上可使用既有的 NumPy、PyTorch 和 JAX 环境。只在调用 AM10 冻结策略时需要 JAX，本地安装选项为 `pip install -e '.[am10]'`。

```sh
policy-learnware prepare-am10 --experiments /share/songyf/RL_Learnware/experiments --out runs/am10
```

每个任务目录的 `audit.json` 保存逐策略的四条来源准入回报和阈值。后四条最终策略自身轨迹专用于上传规约。`source.json` 只收录达标且具有所有目标评分面板的候选；`source_all.json` 包含其余来源达标资产，适合检索耗时测量。真实评分缺少独立参考专家时，`score` 仍计算池内 regret，并将 expert_gap 记为空。来源准入是小样本经验判断。

在 `specify` 上用 `--source-budget`、`--source-prefix` 控制上传观测量。在 `rank` 上用 `--budget`、`--user-prefix` 控制用户观测量，用 `--pool-size`、`--pool-seed` 从来源与算法分层的嵌套子池中取样。核均值分别以各端可见时域的折扣和归一化，长度不同的前缀距离用作召回指标。OPE 只能看到用户提供的转移，人工截短允许在最后一个已知后继状态自举。归档评分保持部署时域 H。有限日志难以识别未观测时域的结果。`--on-invalid abstain` 为默认设置，有非有限入围估值的请求保留失败记录；`--on-invalid raw` 允许在 RKME 召回下转回最近邻，同时记录回退。每个策略保留原部署形态，随机策略用候选专属、由调用者指定种子的随机数流。

```sh
policy-learnware summarize --scores runs/same_setting/*.json --out runs/summary.json
```

每次汇总一个任务、一个实验设置下的评分文件。重复子集种子先按请求求均值。结果分列估值成功数、回退数、返回选择的次数及本地排序时间 P50/P95。计时覆盖用户数据、检索和估值，不含网络传输。归档面板中的最佳候选只用于评分。

### 全池比较

```sh
policy-learnware full-pool --panel panel.json --method wdr --folds 2 --out runs/wdr.json
```

离散面板包含 `request`、`horizon`、`gamma`、`policies` 和 `episodes`。可选 `task`、`split` 用于汇总。候选值是 `[S,A]` 或 `[H,S,A]` 动作分布。每条 episode 提供逐步 `states, actions, rewards` 和可选 `terminated`。短于 H 的记录须明确真正终止。例中状态、动作是整数 ID，两条轨迹是两个独立 episode。

```json
{"request":"example","horizon":1,"gamma":0.9,
 "policies":{"a":[[1,0]],"b":[[0,1]]},
 "episodes":[{"states":[0],"actions":[0],"rewards":[1],"behavior_probs":[[0.5,0.5]]},
             {"states":[0],"actions":[1],"rewards":[2],"behavior_probs":[[0.5,0.5]]}]}
```

| 方法名 | 权限与实施口径 |
|---|---|
| `is, wis, pdis, wpdis` | 整轨迹及逐决策两种加权口径。读取每步真实行为概率 `behavior_action_prob` 或完整 `behavior_probs`，输出 ESS 和零分母诊断。 |
| `dr, wdr, mrdr` | 按完整 episode 交叉拟合本地 critic。DR 和 WDR 使用同一分折和表格 Bellman 拟合，分别用原始和归一化权重校正。MRDR 按方差目标拟合，假定一个 Markov 日志策略；随机候选需要完整行为动作分布。 |
| `dualdice, bestdice` | 时间增广并补零奖励吸收态，要求 γ<1、经验占用覆盖。精确求解表格占用流；BestDICE 的正性与单位质量在有覆盖的精确解上自动满足，二者返回相同点估值。输出普通折扣和、占用比和样本内流残差。不能据此比较两种神经鞍点优化器。 |
| `fqe, am` | 在同一离散面板使用表格 Bellman 回归和经验转移模型前向传播，对齐候选、样本、折扣及回报尺度。充分覆盖时两种精确表格解相同。连续控制仍通过 `rank` 使用原版神经 FQE/AM。 |

另外，`kind="log_probs"` 面板支持连续或离散策略的 `is,wis,pdis,wpdis`。提供 `candidates` 名单，以及每条 episode 的 `rewards`、`behavior_log_prob` 和 `target_log_probs`（候选 ID 到逐步对数概率/密度）。候选概率为零可用 JSON null。使用者需先确认在同一动作测度下存在支撑覆盖，并声明 `common_support=true`；逐步分数本身不能证明全局覆盖。归档的确定性连续 Walker 日志缺少所需权限。两种面板都不读取判分真值，不可估的候选保留原因；全池未能完成比较时，选中策略为空。

原方法来源包括 [DR](https://proceedings.mlr.press/v48/jiang16.html)、[WDR](https://proceedings.mlr.press/v48/thomasa16.html)、[MRDR](https://proceedings.mlr.press/v80/farajtabar18a.html)、[DualDICE](https://papers.neurips.cc/paper_files/paper/2019/hash/cf9a242b70f45317ffd281241fa66502-Abstract.html) 和 [BestDICE](https://arxiv.org/abs/2007.03438)。

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

## 来源策略扩建

在同时提供 PyTorch 与 MuJoCo Playground 的环境里，从随机权重训练一个来源、一个种子。共享服务器已有 `/share/songyf/RL_Learnware/.kmepoa_envs/e01_matrix_py312/bin/python`，可在独立工作树中设 `PYTHONPATH=src`。模拟器和训练器共用 GPU 时设 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，并用 `CUDA_VISIBLE_DEVICES` 选择空闲卡。

```sh
python -m policy_learnware.expansion --task CheetahRun --source s0 --algorithm sac \
  --seed 23 --num-envs 16 --width 128 --batch-size 128 --warmup 1024 \
  --steps 32768 125000 500000 1000000 --out runs/cheetah-s0-sac-r23

python -m policy_learnware.expansion_report runs/cheetah-s0-sac-r23 \
  --out runs/expansion-summary --behavior
```

任务为 `CheetahRun`、`WalkerWalk`、`ReacherEasy`，每个任务有 `s0/s1` 两个归档来源初态分布。时域 1000，action repeat 1。SAC 使用自动温度和双 Q，TD3 使用延迟 actor、target smoothing 和双 Q，PPO-Clip 使用 tanh Gaussian 与 GAE，D4PG 使用同步向量采集、五步回报、51 原子分布 critic 和均匀回放。D4PG 采用同步实现，原论文还研究了分布式异步采集与优先回放。三个离策略训练器默认每个向量采集步更新一次，改变并行环境数会改变每个真实环境步的更新量。四种方法通过确定性 actor 评估和部署。

预算是同一次训练的累计真实环境步数。切片向上对齐到向量批次，PPO 另对齐到完整 rollout，`recipe.json` 保留请求值与实际值。每个切片保存可独立加载的冻结 actor 和验证回报，训练继续沿用原来的优化器及训练状态。按验证均值选一个快照，平分时取较早者，再用 16 个独立回合与来源门槛比较。通过后生成独立的规约轨迹和共同保留初态上的行为诊断轨迹。四组数据分别用于选择、复核、规约、行为比较。训练折扣为 0.99，准入使用未折扣的完整回合累计奖励。单次运行的输出目录需为新目录，快照用于部署和评估。

每次传入一个或多个独立运行目录给 `expansion_report`。汇总中的单卡训练小时包含未达标训练，GPU 训练小时与整次训练及评估的墙钟时间分别计量。输出的 `<任务>/source.json` 可直接传给 `policy-learnware specify` 并配用同任务的 settings。可选行为比较按相同来源和接近的复核回报组成策略对，报告动作 MSE、两组保留轨迹各自的占用 MMD，并按控制时间段记录 Cheetah 向前速度。
