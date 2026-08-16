# Rescue Drone 因果环境设计审计与更新路线

日期：2026-08-16
用途：2026-08-17 lab meeting 设计评审
状态：meeting-ready prototype；不是 publication-ready benchmark

## 1. 结论先行

本次检查得到四个核心结论。

1. **当前 Rescue Drone 基线无法在可访问代码中定位。** 已执行 `git fetch --all --prune`，检查 `Causal-Gymnasium` 所有本地与 `origin/*` 引用，并通过 GitHub 搜索仓库代码、分支和 PR；截至 2026-08-16，没有包含 `rescue` 或 `drone` 的文件、分支或 PR。因此，本报告不能诚实地声称完成了“旧 Rescue Drone 代码逐行 diff 审计”。本次交付是依据教材和最新版 Lunar Lander 架构建立的 **provisional reference design**。一旦原 Rescue Drone branch/PR 可见，应补做一次差异审计。
2. **Lunar Lander 适合作为工程架构模板，但不是可以原样复制的理论金标准。** 它清楚地分离了 `LunarLanderSCM` 与 `LunarLanderPCH`，并提供 `see/do/ctf_do`、图接口和 notebook；但 latent wind 的生命周期、隐藏信息暴露、图与实际机制的一致性、`do()` 的 callable 契约，以及 `ctf_do()` 的理论边界仍有明显风险。
3. **新的 Rescue Drone 草稿把“因果性”落实到了可执行机制。** 每个 stage 只采样一次外生变量 `U_i=(gust_i, hazard_i)`，自然动作、状态转移和 reward 共用同一 `U_i`；`do()` 只替换动作机制；一阶段的 paired counterfactual 对所有候选动作复用同一个 state 和同一个 `U_i`。
4. **当前应展示“理论正确的最小环境”，而不是宣称完成了最终 benchmark。** 明天最稳妥的 meeting 结论是：我们已经形成一个教材可追溯、接口可运行、测试可复现的设计基线；下一步需要 PhD 确认科学问题、estimand、reward、hidden information 和环境复杂度。

## 2. 审计证据与版本基线

### 2.1 Git / GitHub 证据

- 同步后的 `origin/main`：`c94e9a4`，2026-08-03。
- Lunar Lander 环境的最新 PhD 作者提交：`f2ce211`，作者 `ml`，2025-06-13，主题 `add where to intervene support`。
- Lunar Lander notebook 的最新 PhD 作者提交：`0d2ca8d`，作者 `ml`，2025-10-24，主题 `update readme and fix minor API issues`。
- 当前工作分支：`ey/causal-semantics-fixes`。
- Rescue Drone 搜索结果：本地文件、所有 Git refs、GitHub code、branch、PR 均为 0 个匹配。

因此，“目前 Rescue Drone 设计”的唯一可验证状态是：**它没有出现在当前 checkout、官方远端分支或可访问 PR 中**。这不是说设计一定不存在，而是说本次审计没有可追溯的原始 artifact。

### 2.2 教材基线

理论检查只使用工作区中的 [Causal AI 教科书](../../../../Causal%20AI%20Reading/Causal%20AI.pdf)，版本信息为 Draft v0.27, October 2025。下文页码为书内印刷页码；括号内给出 PDF 页码，便于复查。

| 教材锚点 | 对环境设计的要求 |
|---|---|
| Definition 2.1.1, SCM, p.44（PDF p.56） | 必须明确 `M=<U,V,F,P(U)>`；不能只有变量名和一张图。 |
| Definitions 2.2.1-2.2.5, PCH L1/L2, pp.46-49（PDF pp.58-61） | `see` 由原机制生成；`do(x)` 替换动作机制且保持其余机制与 `P(U)` 不变；hard intervention 必须满足 effectiveness。 |
| Definition 2.2.6, PCH L3, p.51（PDF p.63） | 多个 counterfactual world 必须由同一个 unit `u` 评价；换 action 时不能重采样外生现实。 |
| Definition 2.4.1, causal diagram, pp.61-63（PDF pp.73-75） | directed edge 来自函数参数；bidirected edge 来自共享/相关外生变量。图必须由代码里的机制推出。 |
| Example 7.2, MDP as SCM, pp.517-518（PDF pp.529-530） | 序列环境应显式写出 state、action、reward 和 stage noise 的结构方程。 |
| Definition 7.2.1 and Example 7.6, pp.521-522（PDF pp.533-534） | policy intervention 替换行为策略 `f_X`，不是另开一套 transition/reward simulator。 |
| Definitions 8.1.1-8.1.4, pp.530-539（PDF pp.542-551） | 完整 decision task 还必须定义 policy space、reward function、learning regime 和 structural assumptions。 |
| Definition 8.1.5 and Examples 8.10-8.13, pp.542-546（PDF pp.554-558） | Markov property 本身不能把 observational transition 当作 interventional transition；必须说明数据来自 `see` 还是 `do`。 |
| Definition 8.2.1, NUC, p.547（PDF p.559） | 如果 action 和 outcome 共享未观测外生来源，就不能宣称 NUC。 |
| Definition 9.4.4, confounded MDP environment, p.623（PDF p.635） | 可用 `S_i, X_i, Y_i, U_i` 的 stage-wise SCM 建模 MDPUC；若 `U_i` 持久且未进入 state，需重新判断 Markov/POMDP 边界。 |

教材 Example 7.2/7.6 最重要的启示不是某个 XOR 公式，而是：同一个外生误差可以同时进入动作与 outcome；执行 policy intervention 时只替换动作方程。Examples 8.10-8.13 进一步说明，即使 observational 和 interventional worlds 都满足 Markov property，它们也可能对应不同 transition/reward 与不同最优 policy。

## 3. Lunar Lander 架构审计

### 3.1 值得继承的工程结构

| 结构 | 位置 | 价值 |
|---|---|---|
| `SCM` / `PCH` 双类分层 | `causal_gym/envs/lunar_lander.py:33`, `:215` | 世界机制与 agent interaction capability 分离。 |
| 显式 `sample_u()` | `:70-76` | 给外生随机性一个独立入口。 |
| Gym `reset/step` | `:90-116` | 与现有 Causal-Gymnasium、Gymnasium 工具兼容。 |
| `action()` / `observation()` | `:119-132` | 明确 natural behavior policy 与 learner observation。 |
| causal graph API | `:186-209` | 可向算法与 notebook 暴露结构假设。 |
| `see/do/ctf_do` | `:227-250` | 与 repo 的 PCH interaction vocabulary 对齐。 |
| render + notebook | `:134-181`, `examples/test_lunar_lander.ipynb` | 适合 meeting 演示和人工验证。 |

### 3.2 不能原样复制的理论/实现风险

1. **latent wind 被 `reset()` 直接放入 `info['wind_map']`。** `lunar_lander.py:93` 与“latent/unobserved”叙述冲突。Gym 的 `info` 可以服务 debug，但如果 learner 能直接读取，它就不再是未观测 confounder。建议默认隐藏，只在显式 debug 模式下暴露。
2. **natural action 与 current wind 的时间顺序不稳定。** `see()` 在 `:231` 先调用 `env.action()`；但 local wind 在 `step()` 的 `:99-102` 才按当前位置更新。第一步 policy 看到 `None`，后续可能看到上一步 wind。这样 `f_X` 与 `f_S` 未必使用同一个 `U_i`。
3. **graph 与函数参数没有完全对齐。** Wind 影响 physics transition，并间接影响 reward；behavior policy 也读取 wind。当前图注释掉显式 `U`，只保留 `X <-> S'`，没有完整表达 shared exogenous source 对 `X/Y/S'` 的关系。
4. **episode-level wind map 改变任务类别。** 一个固定 wind field 在多个 stage 持续存在。如果 learner state 不包含这个 field 或足够 belief state，任务更接近 POMDP；不能只因为底层 Gym 是 LunarLander 就自动称为 standard MDP。
5. **`do()` 的契约是 policy callable，不是 raw action。** `LunarLanderPCH.do()` 在 `:237-239` 调用 `do_policy(observation)`；原 notebook 曾传入 NumPy integer，导致 `'numpy.int64' object is not callable`。Rescue Drone 草稿显式检查 callable 并给出可行动的错误信息。
6. **`ctf_do()` 不是完整的 retrospective L3 engine。** 现有实现会读取 natural intention 后立即替换 action，适合 Chapter 9 的 intention-conditioned counterfactual intervention；但它没有保存事实 world、abduct `U`、再同时重演多个 alternative worlds。不能把一次 `ctf_do()` 直接描述为已解决任意 L3 query。
7. **`Task.assumptions` 枚举缺少 `semi_markovian`/`admg`。** Repo 目前只有 `dag`, `markov`, `nuc`，而共享 `U_i` 的 latent projection 含 bidirected edges。草稿 notebook 暂用 `dag` 只是现有 API 限制，不是严格理论标签；这是 core API 的 P0 待办。

结论：Lunar Lander 是 **role-model scaffold**；Rescue Drone 应继承其模块边界和演示方式，但需要重写 exogenous lifecycle、graph derivation、hidden-information contract 和 L3 validation。

## 4. Proposed Rescue Drone SCM

### 4.1 任务语义

单架 drone 在有限网格中寻找一个 victim。每一步需要在电池耗尽或 victim health 归零前到达 victim cell。Learner 能看到 drone/victim 位置、电池、health，但看不到当步 gust 与 hazard。自然 pilot 有本地传感器，能感知 gust；因此 logged natural actions 可能与 outcome 受到共同的未观测原因影响。

### 4.2 SCM 形式化

对每个 stage `i`：

- `U_i = (G_i, H_i)`：gust direction 与 hazard severity，来自 `P(U_i)`；当前草稿设为跨 stage i.i.d.。
- `S_i`：drone position、battery、victim position、victim health、rescued flag。
- `O_i <- f_O(S_i)`：learner observation；不含 `U_i`。
- `X_i <- f_X(O_i, U_i)`：natural pilot action。
- `S_{i+1} <- f_S(S_i, X_i, U_i)`：移动、gust drift、energy cost、health decay 和 terminal state。
- `Y_i <- f_Y(S_i, X_i, U_i)`：rescue/failure outcome 与小幅 progress shaping。

因此：

`M = <U={U_1,...,U_H}, V={S,O,X,Y}, F={f_O,f_X,f_S,f_Y}, P(U)=prod_i P(U_i)>`。

这与教材 Definition 9.4.4 的 stage-wise confounded MDP construction 对齐；加入 `O` 是为了把 learner information set 明确写出来。

### 4.3 PCH 语义

- `see()`：采样一次 `U_i`，计算 natural action `X_i=f_X(O_i,U_i)`，再用同一个 `U_i` 计算 transition 和 reward。
- `do(policy)`：采样一次 `U_i`，用 `policy(O_i)` 替换 `f_X`，其余机制和 `P(U)` 不变。传入的必须是 callable。
- `ctf_do(policy)`：先用 `f_X(O_i,U_i)` 得到 intended action，再由 `policy(O_i,intended)` 选择 realized action；transition/reward 继续使用同一个 `U_i`。
- `unit_counterfactuals()`：从同一 `S_i,U_i` 评价所有 action alternatives；这是 full-simulator one-step L3 diagnostic，不是从 logged data 自动识别 counterfactual。

### 4.4 图结构

由函数参数推出 directed edges：

- `S -> O -> X`
- `S -> Y`, `X -> Y`
- `S -> S'`, `X -> S'`

由于 `U_i` 同时进入 `f_X, f_Y, f_S`，latent projection 还必须包含：

- `X <-> Y`
- `X <-> S'`
- `Y <-> S'`

这三条 bidirected relationships 不是装饰，而是 Definition 2.4.1 对 shared exogenous arguments 的直接结果。

## 5. 覆盖程度

### 5.1 Theory coverage

| 检查项 | 草稿状态 | 结论 |
|---|---|---|
| `U,V,F,P(U)` 明确 | Covered | 代码 docstring、`sample_u`、纯 transition/reward function 对齐。 |
| L1 `see` | Covered | natural policy 与 transition 共用 stage noise。 |
| L2 `do` mechanism replacement | Covered | 只替换 action rule；callable contract 有测试。 |
| Effectiveness | Covered at API level | `info['action']` 与 forced policy 输出一致；仍建议增加 property-based tests。 |
| L3 same-unit semantics | Covered for one step | 所有 alternatives 共用 `S_i,U_i`；long-horizon abduction/replay 未实现。 |
| Causal graph from functions | Covered | directed/bidirected edges 有测试。 |
| Policy space | Partial | action/observation space 已定义；`Task.policy_space` 尚未填充。 |
| Reward function | Partial | executable，但 scientific objective 尚未由 PhD 确认。 |
| Learning regime | Covered | `see/do/ctf_do` permissions 使用 repo `Task`。 |
| Structural assumption label | Gap | 当前 enum 无 `semi_markovian/admg`。 |
| Markov/POMDP boundary | Covered for i.i.d. draft | 若升级 persistent wind map，必须重新设计 state/belief。 |
| Identifiability claim | Intentionally absent | 没有在未指定 estimand/assumptions 时过度声称。 |

### 5.2 Engineering coverage

| 检查项 | 草稿状态 | 结论 |
|---|---|---|
| Gym reset/step API | Covered | seeded reset、5-return step。 |
| Deterministic reproducibility | Covered | 同 seed 的 state/noise/outcome 一致。 |
| Hidden-variable leakage | Covered by default | 只在 `include_latent_in_info=True` 时暴露。 |
| PCH facade | Covered | 参考 Lunar Lander 的 class split。 |
| Rendering | Covered | headless `rgb_array`。 |
| Notebook | Covered | fresh-kernel 端到端执行成功。 |
| Unit tests | Covered for core semantics | 6 个定向测试通过。 |
| Packaging/registration | Intentionally deferred | 草稿尚未加入 `envs.__init__` 或 Gym registry。 |
| Algorithm benchmark | Missing | 尚无 RL/causal-RL baseline、confidence interval、sample complexity。 |
| Original Rescue branch diff | Blocked by source gap | 需要 branch/PR/path。 |

## 6. 本次草稿的 code/theory 修改分离

### 6.1 Code layer

- 新增 `causal_gym/envs/rescue_drone.py`：`RescueDroneSCM` + `RescueDronePCH`。
- 新增 stage-noise pending lifecycle，保证每步只采样一次。
- 新增 default hidden-info policy；latents 默认不进入 `info`。
- 新增 pure `_transition()`，支持同 unit alternatives。
- 新增 `unit_counterfactuals()`、`rgb_array` render 和 causal graph。
- 新增 `tests/test_rescue_drone_draft.py`，覆盖 seeding、hidden leakage、L1/L2/L3 interaction、callable contract 与 graph。
- 新增 `examples/test_rescue_drone.ipynb`，按 Lunar Lander 的“环境创建 - regime demo - render”结构组织。

### 6.2 Theory/result layer

- 将环境正式写成 `M=<U,V,F,P(U)>`，而不是“gridworld + causal label”。
- 将 confounding 实现为 shared `U_i`，而不是只在图里加 bidirected edge。
- 将 observational 与 interventional data-generating regime 分开。
- 将 L3 声明限制为 one-step same-unit simulator comparison，不夸大成任意 counterfactual identification。
- 将 persistent weather 明确列为后续 POMDP/augmented-state 设计选择。
- 将当前结果解释为接口与语义 smoke evidence，不解释为 policy superiority。

## 7. 修改建议与更新方向

### P0：meeting 后立即确认

1. **找回原 Rescue Drone artifact。** 需要 branch、PR、fork URL 或文件路径；随后做旧实现 vs 本草稿的逐行差异审计。
2. **确认 scientific question / estimand。** 候选：`E[Y | do(X=x)]`、policy value、ETT `E[Y_x | X=x']`、success probability、time-to-rescue 或 regret。没有 estimand，就无法判断哪些 causal features 是必要的。
3. **确认隐藏信息。** 自然 pilot 到底能看到 gust、thermal signal、victim severity 中的哪些？Learner 能看到哪些？这决定 `f_X`、policy space 和 confounding。
4. **确认 task category。** i.i.d. gust 可维持当前 stage-wise MDPUC；persistent wind field 若不进入 state/belief，则应明确称为 POMDP。
5. **扩展 `Task.assumptions`。** 增加 `semi_markovian` 或 `admg`，避免用 `dag` 标签承载含 bidirected edge 的模型。

### P1：形成可合并环境

1. 把 approved class 导出到 `causal_gym/envs/__init__.py` 并注册 Gym ID。
2. 把 `U_i` lifecycle 抽成 base-SCM contract：`prepare_u -> action/transition/reward -> consume_u`。
3. 标准化所有环境的 `do(policy)` 与 `ctf_do(policy)` callable signature，并统一错误信息。
4. 加 graph-mechanism consistency tests、effectiveness property tests、no-latent-leak tests。
5. 定义 `Task.policy_space` 和正式 reward function；删除未经确认的 shaping 或把 shaping 分项放入 `info`。
6. 为 `see` 和 `do` 生成分开的 dataset schema，明确 regime metadata。

### P2：研究级 benchmark

1. persistent spatial wind + belief state / history-based policy。
2. multi-victim、triage、no-fly zones、sensor failures 与 heterogeneous environments。
3. observational/offline、online intervention、mixed policy 和 counterfactual-policy baselines。
4. paired seeds、confidence intervals、ablation（无 confounding / 有 confounding / latent revealed）。
5. 预注册主要 metric 与 failure taxonomy，避免根据 demo 结果事后改 reward。

## 8. 明天 slide show 的推荐叙事

1. **问题：** 一个 rescue game 什么时候才是 causal environment？
2. **教材标准：** `SCM -> PCH -> CDM/CRL task`，图必须来自机制。
3. **Lunar Lander role model：** 工程结构优秀，但理论 contract 需要显式化。
4. **Rescue Drone model：** 一页讲清 `U,S,O,X,Y,S'` 与函数。
5. **三种 interaction：** see 保留 natural action；do 替换 action；ctf-do 保留 intention 与同一 `U`。
6. **代码架构：** `Agent -> PCH -> SCM`，SCM 内部每 stage 只采样一次外生变量。
7. **证据：** 6 tests passed + notebook fresh-kernel passed。
8. **诚实边界：** 原 branch 未定位；reward、estimand、POMDP 选择未定。
9. **meeting 决策：** 让 PhD 对 P0 的五个问题作出选择。

## 9. 验证记录

### Environment tests

```bash
.venv/bin/python -m pytest tests/test_rescue_drone_draft.py -q
```

结果：`6 passed`。

### Fresh-kernel notebook

验证时创建了仅位于 `/tmp/rescue-drone-jupyter` 的临时 `.venv` kernelspec，没有修改全局 Jupyter 配置。

```bash
JUPYTER_PATH=/tmp/rescue-drone-jupyter/share/jupyter \
  .venv/bin/python -m jupyter nbconvert \
  --to notebook --execute examples/test_rescue_drone.ipynb \
  --output /tmp/test_rescue_drone.executed.ipynb \
  --ExecutePreprocessor.timeout=180 \
  --ExecutePreprocessor.kernel_name=rescue-drone-venv
```

结果：执行成功。30 个 matched seeds 的 smoke demonstration 输出为：

- `see`: success `1.00`, mean return `0.967`
- `do`: success `1.00`, mean return `0.960`

这两个数只说明两个 regimes 都能稳定运行；差异太小，**不能**作为 natural policy 优于 interventional policy 的统计结论。

## 10. Meeting 中建议直接问 PhD 的五个问题

1. 原 Rescue Drone 设计在哪个 branch/PR/fork？
2. 主要 causal query 是 policy value、ATE、ETT、where-to-intervene，还是 counterfactual policy？
3. natural pilot 与 learner 各自能看到哪些变量？
4. 风场是每步 i.i.d.、整局固定，还是空间固定？
5. reward 的最终优先级是 rescue success、time、energy、risk，还是加权组合？

这五个答案将决定下一版是“confounded MDP benchmark”、POMDP、mixed-policy environment，还是 counterfactual-policy environment。
