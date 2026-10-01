# PINNForge

[English](README.md) | **中文**

PINNForge 用自主编程智能体来设计物理信息神经网络（PINN）。一个 **block** 就是一个智能体、一份固定的 GPU 时间预算和一份书面总结。block 依次串行运行，经验通过总结文件传递，而不是存在编排器的状态里。每个候选方案都按相对 L2 误差（rRMSE）与参考解比较打分，参考解由一个评分服务持有，智能体读不到。编排器是一个普通的 Python 进程：一个 block 是否完成，看文件系统就能判断，不需要让模型来决定。

## 安装

```bash
curl -fsSL https://raw.githubusercontent.com/JiangtaoShen/pinnforge/main/install.sh | sh
```

脚本通过 `uv tool install` 从本仓库把一个轻量的 `pinnforge`（只依赖 PyYAML、NumPy）装到 PATH 上。不要用 `pip install pinnforge`：PyPI 上的 `pinnforge` 是另一个无关的项目。目标平台是带 NVIDIA GPU 的 Linux；macOS 和纯 CPU 机器也能运行，只是慢。Windows 请在 WSL2 中使用。

## 五条命令

```bash
pinnforge init ~/pf && cd ~/pf     # 创建工作区：选择使用哪些 GPU，并建立固定版本的 JAX 环境
pinnforge demo                     # 用脚本化智能体在几分钟内跑完一次完整实验，不需要 API key
pinnforge setup                    # 选择并实测负责运行 block 的编程智能体
pinnforge task new                 # 描述一个 PDE；没有参考解时由智能体构建
pinnforge run start -t <task> -n 5 # 在后台运行，并打开实时面板
```

`pinnforge doctor` 会检查所有前置条件，每个问题都给出修复命令。`pinnforge ui` 打开实时面板，内容包括：每次评估的分数随 GPU 时间的变化（带迄今最佳曲线和 b00 基线）、带实时预算进度的 block 时间线、以动画回放的迄今最佳预测场、正在运行的智能体的实时工具调用。通过 SSH 使用时：`ssh -L 8765:127.0.0.1:8765 <主机>`。

## 任务，以及标准答案从哪里来

```bash
pinnforge task new burgers --describe "u_t + u u_x = 0.01/π u_xx on [-1,1]×[0,1],
  u(x,0) = -sin(πx), u(±1,t) = 0; score u"                  # 参考解由智能体构建并经过检验
pinnforge task new ldc2 --describe @spec.md --reference ~/data/ldc.mat   # 使用你自己的参考解
pinnforge task status burgers --follow
```

`task new` 是一条分阶段的流水线，进度保存在 `tasks/<name>/.authoring/`：

1. 任务编写智能体写出任务包（`problem.md`、`baseline.py`、`eval.py` 和公开的评估网格），全程看不到答案。
2. 参考解要么从你提供的数据导入，要么由专门的**参考解构建智能体**用经典数值方法在独立环境中、在至少三个分辨率上计算。
3. 框架亲自检验参考解：用任务自己的指标拿每一层与下一层比较，测出收敛阶，估计最细一层的误差，再重跑最粗的两层证明结果可复现。
4. 契约校验、免费的 CPU 冒烟测试、测量 b00 基线。

任何阶段失败，修正后用 `--resume` 接着跑；此时给出的参数（如 `--ref-budget`、`--accept-reference`）会覆盖之前记录的值。`--redo STAGE` 会重跑某个阶段及其后的所有阶段，例如收敛检验失败后，用 `--resume --redo reference` 重新构建参考解。如果知道精确解，请把它作为参考解提供（写有公式的 `.py` 或 `.txt` 文件），不要写在问题描述里，因为任务编写智能体会读到描述。

参考解的任何信息都不会进入交给 block 的任务包：没有数值，没有解析式，也不说明它是怎么得到的、精度如何。具体做法：

- 任务编写智能体工作期间，本任务的 `private/` 会被移出它的工作目录；结束后框架审计它的每一次工具调用，只要碰过任何任务的答案，它写出的任务包就会被移进 `private/logs/` 并重新编写。
- 参考解导入智能体和参考解构建智能体看过答案，它们写的一切都留在 `private/` 里：写到别处的东西会被框架移进 `private/logs/`，该阶段判为失败；它们所用工具自身保存的会话记录（Claude Code、Codex）也会被移到那里。
- `task new` 遵守 `private_dir` 的设置，答案放在它指定的位置。
- 建任务记录和实时面板只拿到框架在检验中自己算出的数字。
- `task validate` 会检查每个任务包：是否提到了解（精确解、解析解等），是否引用了答案所在的路径，公开数据里是否有参考解的拷贝，公开文本里是否出现了参考解的数值。

`pinnforge examples add ldc ks cyl naca` 会下载四个基准任务及其参考解（校验 sha256）；演示任务随安装包附带。

## 运行

```bash
pinnforge run start -t ldc -n 5 --agent opus    # 使用 `pinnforge setup` 创建的绑定
pinnforge run start -t ks --gpus 1 --budget 3600
pinnforge run attach ldc_1                      # 进入驱动进程的终端（Ctrl-b d 退出且不中断）
pinnforge run status ldc_1                      # 每个 block 的状态、GPU 秒数、评估次数、总结、最佳分
pinnforge run stop ldc_1                        # 像 Ctrl-C 一样有序停止
pinnforge run resume ldc_1
pinnforge run audit ldc_1                       # 检查是否有 block 改动了别人的记录
```

`run start` 把任务和设置冻结到 `runs/<task>_<n>/`，安装 b00 基线（来自缓存、以前的运行，或现场测量），然后在一个独立的 tmux 会话里逐个派发 block，所以关掉终端也不影响运行。

- 一个运行只由一个进程驱动（`runs/<id>/.lock`）。
- SIGTERM、SIGHUP 与 Ctrl-C 走同一条收尾路径：中断智能体的整个进程组，记下这一段的开销，写入完整性结论。
- `run stop` 是给这个进程发信号，而不是改它的状态文件。
- 一个 block 只有在总结已写、预算已用完、并且至少有一次真实评估时，才算完成。

**GPU。** 工作区的运行使用哪些卡由用户决定：机器上有多张卡时，`pinnforge init` 会列出每张卡及其当前占用情况并询问（也可以直接用 `--gpus 1`、`--gpus 0,1`、`--gpus none` 指定）；以后用 `pinnforge setup gpus` 修改。选择保存在 `pinnforge.yaml` 的 `sandbox.gpus` 中，可以是：

- `auto`：使用 `nvidia-smi` 列出的所有卡，并受 `CUDA_VISIBLE_DEVICES` 限制；由调度系统（如 SLURM）分配卡时应选它；
- 一个明确的列表，如 `[1]`、`[0, 2, 3]`：只用这些卡；
- `[]`：只用 CPU；
- `null`（新工作区在没有人选卡时的取值）：使用本机的默认值（`pinnforge setup gpus --user` 设置），没有则为 `auto`。

一张卡时，block 的评估在这张卡上排队；多张卡时，block 可以在这些卡上并行评估。`run start --gpus` 只对这一次运行临时改用别的卡。如果某张卡已被另一个运行占用，`run start` 会拒绝启动（加 `--share` 仍可共用）；`demo` 只使用空闲的卡。

## 智能体

| runtime | CLI | 默认模型 |
|---|---|---|
| `claude_code` | `claude` | `claude-opus-4-8` |
| `codex` | `codex` | `gpt-5.4` |
| `cursor_agent` | `cursor-agent` | `auto` |
| `opencode` | `opencode` | 无，需要指定 |

`pinnforge setup` 为每个 CLI 固定一个**精确的**模型 id，并通过与 block 相同的适配器实测连通。之所以要精确 id：`opus` 这个别名曾经一年里指 `claude-opus-4-8`，之后又指 `claude-opus-5`，两个相隔一代、都写着 "opus" 的运行其实没有可比性。无论请求的是什么，框架都会把智能体实际运行的模型 id 读回并记入运行账本。

仓库没见过的智能体 CLI：把 runtime 设为 `module.path:ClassName`，实现 `pinnforge/runtime/base.py` 中的 `AgentRuntime` 协议即可。

## 论文笔记库

block 在选定方向之前会先读 `kb1/`：随安装包附带的 130 篇论文笔记，每篇的 Method 部分都有可以直接运行的 JAX 代码，并会复制进每个运行。`pinnforge kb1 add --source <论文>` 会把更多论文提炼进工作区自己的 `kb1/`（首次使用时先复制一份随包语料）；`pinnforge kb1 check` 检查格式和大小。

## 用对话来使用

`plugin/` 目录让你从自己的智能体里驱动 PINNForge。它包括：

- 一组 skills；
- **task-author** 子智能体：把关于某个 PDE 的对话变成一个可用的任务；
- **run-doctor** 子智能体：只读地诊断卡住的运行；
- 一个会话钩子。

```text
/plugin marketplace add JiangtaoShen/pinnforge        # Claude Code
/plugin install pinnforge@pinnforge-marketplace
```

也可以直接运行 `pinnforge chat`，它会打开 Claude Code 并在本次会话中加载插件。Codex、Cursor、OpenCode 请按项目安装插件——装在你自己的项目目录里，不要全局安装：block 也由这些 CLI 运行，而只有 Claude Code 的 block 会关闭插件。详见 `plugin/README.md`。

不要把 `AGENTS.md`、`CLAUDE.md` 或 `.cursor/rules/` 放在工作区里或它的上级目录：block 在 `runs/<id>/` 中运行，会读到它们。

## 评分与记录

`eval.py` 在公开网格上训练和预测，把预测结果提交到队列；评分服务把预测与参考解比较，再把分数写回。评分服务：

- 只返回 rRMSE 这一个数；算不出时只返回一个原因，而这个原因只依据提交的内容和公开网格判断。任务指标算出的其他量（比如 MSE，它和 rRMSE 放在一起就能反推出参考解的范数）一律不外传，指标自己的报错信息只写进该任务的 `private/logs/`；
- 从不执行智能体代码；
- 从自己保存的任务快照中一次性加载指标；
- 从不跟随队列中被预先放置的符号链接。

答案存放在 `tasks/<t>/private/`，永远不会复制进运行目录。如果要让隔离成为操作系统层面的事实，把答案放到工作区之外（在 `pinnforge.yaml` 中设置 `private_dir:`，或在 `~/.config/pinnforge/config.yaml` 中为所有工作区设置），并以另一个系统用户运行评分服务：`pinnforge scored <run> --private /srv/answers`。

每次派发前后，框架都会对 block 不得改动的内容计算哈希：之前的评估和总结、任务包、charter、论文笔记库。结论写入 `runs/<id>/.integrity/`，供 `run audit` 读取。这是有意选择的"检测"而不是"阻止"：真要做隔离，就得列出 block 合法写入的每一条路径。

在默认的同用户模式下，一个存心作弊的 block 仍然*读得到*任务库里的答案；上面那种分用户部署才能堵上这一点。

实时面板是只读的，本机任何进程（包括 block）都能访问。因此它：

- 只返回由已知文件构造的 JSON，从不读取 `private/`、评分队列、参考解或误差场；
- 校验 `Host` 请求头；
- 在监听非回环地址时要求 token。

## 任务契约

框架只通过以下接口接触任务。

* **`problem.md`**：block 唯一会读的说明文字，固定七个章节。
* **`baseline.py`**：根候选方案。包含冻结的 `PDE CONSTANTS` 文件头，以及 `train(rng, eval_callback=None) -> (params, step_count)` / `predict_fn(params, X) -> dict` 两个函数。
* **`eval.py`**：任务自己的常量、网格处理（`INPUT_COLUMNS`、`FIELDS`、`REF_COLUMNS`、可选的 `predict_rows`）和评分指标 `score_predictions(fields, ref_dir)`，它只返回 `{"rRMSE": x}`。
  - 通用机制（预算、逐卡 GPU 锁、worker 子进程、评分队列、记录格式）是共享的，会以 `task/_forgekit.py` 的形式复制进每个运行，和 `task/` 里的其他文件一样冻结并计算哈希。
  - 用法：`eval.py blocks/bNN/<file>.py [--gpu G] [--seed S] [--smoke | --diag]`。每次运行向 `blocks/bNN/evals.jsonl` 追加一条记录；只要写入了记录就以 0 退出。
* **公开数据**放在任务目录里，**答案**放在 `private/` 里。

`pinnforge task validate <name>` 运行契约检查。

b00 基线按任务缓存，以任务包指纹、参考解和硬件类别为键。判断是否过期要看这个键，而不是看分数：训练按墙钟时间截止，同一个任务重测，分数会有 0.4%–17% 的波动。

`task anchor --rekey` 可以不花 GPU 时间就沿用基线，前提是用缓存的参数重新评分能精确复现记录的分数。

## 工作区结构

| 路径 | 作用 |
|---|---|
| `pinnforge.yaml` | 工作区标记，以及它的设置（智能体、GPU、预算、private 目录、面板） |
| `pyproject.toml`、`.venv/` | 冻结的 JAX 环境，每个 `eval.py` 都在其中运行 |
| `tasks/<name>/` | 任务库 |
| `runs/<task>_<n>/` | 一次运行：冻结的任务副本、论文笔记副本、`blocks/bNN/`、`blocks/kb2/`、账本、日志 |
| `.pinnforge/` | 锁、评分账本、参考解智能体的求解器环境、面板状态 |

设置的优先级从低到高：内置默认值 < `~/.config/pinnforge/config.yaml` < `pinnforge.yaml` < 命令行参数 < `--set a.b=c`。某个键在文件中为 `null` 或不存在，就表示该文件没有设置它，由下一层决定；新建的 `pinnforge.yaml` 中 `agent`、`sandbox.gpus`、`sandbox.wall_budget_s` 和 `private_dir` 都是 `null`，因此在工作区自己设置之前，本机配置文件对所有工作区生效。`FORGE_PRIVATE_DIR` 优先于这两个文件。每次运行都会把实际使用的设置冻结到自己的 `run.yaml` 中。

## 从 0.1 升级

四个基准任务的 `eval.py` 现在把通用机制交给共享的 kit，这会改变它们的指纹。用 `pinnforge task anchor <task> --rekey` 可以不花 GPU 时间就沿用各自缓存的基线：它会用缓存的参数重新评分，要求得到完全相同的数值。

升级之前创建的运行里冻结的是旧版 `eval.py`，拿新的任务库去续跑，会被判定为指标漂移并拒绝，除非设置 `PINNFORGE_ALLOW_METRIC_DRIFT=1`；即使设置了，旧版 `eval.py` 发出的评分请求也不注明当前 block，新的评分服务不会受理。请用 0.1 把旧运行跑完，或者直接开新的运行。

## 开发

```bash
git clone https://github.com/JiangtaoShen/pinnforge && cd pinnforge
uv sync                       # 开发工具加冻结的 JAX 环境；源码目录本身就是一个工作区
uv run pytest tests/ -q       # 快速测试
uv run pytest -m slow         # 真实评估（需要 JAX、GPU 和 private/）：eval.py 行为一致性测试
uv run ruff check .
```

`tasks/` 和 `pinnforge/examples/` 有意不做 lint：它们的 `baseline.py` 和 `eval.py` 会被基线缓存计算指纹，一次纯空白字符的修改就会让所有缓存的 b00 失效。它们改由 `pinnforge task validate` 把关。
