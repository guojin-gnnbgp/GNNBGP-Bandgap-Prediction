# GNNBGP Industrial Workbench v0.6 · 三算法 GPU 联调与内嵌报告


## v0.6 · 三算法 GUI 联调（以本次上传源码为基础）

**本版本不会修改算法、甲方 `gnnbgp` 源码、PyTorch 环境，也不会重新编译 ABACUS。**
当前 GUI 已能导入 XYZ；保留材料导入、结构视图、历史任务、原位计算报告及归档删除。

本次补齐的是**运行环境衔接**：

1. 启动脚本自动识别已编译的 GPU ABACUS（默认 `~/BandGap-Studio/build/abacus-install/bin/abacus_pw_gpu`）。
2. 自动将 `~/bgp-deps/lib` 加入 `LD_LIBRARY_PATH`（保留已有 CUDA/MPI/UCX 路径），不执行 `conda activate`。
3. `/api/system` 报告 ABACUS 路径、是否可执行、依赖库目录是否存在；前端默认填入该路径，不覆盖用户手动输入。
4. 提交功能 2/3 时明确执行 `bgp predict ... --backend <算法> --abacus <可执行文件> --work-root <Linux目录>`。未配置有效 ABACUS 时，提交前给出错误；GNN 不受影响。
5. 每个任务使用隔离的 Linux 运行目录 `workspace/jobs/<id>/engine-runs`，规避 WSL `/mnt/d` 下 `input/` 和 `INPUT` 大小写冲突。
6. 界面报告继续从真实 `result.json` 展示，不伪造运行结果。

**重要：测试中的 `fake_bgp.sh` 仅模拟 CLI/HTTP 调用。14+ 单元测试通过不代表在你的物理 GPU 上进行过科学计算。** 真实验收必须在本机 WSL 中分别提交 GNN-HSE06、CIDER-PBE0、ABACUS-HSE06 并核对 `result.json`。

详细步骤见 [GUI_三算法联调指南_v0.6.md](GUI_三算法联调指南_v0.6.md)。

---

## v0.5 功能增强（保留 v0.4 工业界面与真实计算适配）

### 1. 已导入结构增加删除按钮

在左侧 **「已导入结构」** 中，每个已导入文件右边增加红色删除（垃圾桶）按钮。
点击后确认，调用 `POST /api/materials/<材料ID>/remove`。

- 从未用于计算的材料：从工作区列表和对应副本中删除；
- 已参与计算的材料：从当前列表移除（归档），原始导入副本保留，历史任务、结果和“重复运行”功能不受损；
- `D:\program\gnnbgp` 内的模型、源码及 `examples/` 示例不会被修改；
- 用户原有数据库自动进行非破坏性的 `archived_at` 列迁移。已归档材料不在工程树显示。

### 2. 三大业务计算类型

右侧 **「计算类型」** 下拉框严格映射甲方源码中真正支持的四个 CLI 后端：

| 计算类型 | 具体计算方法 | 后端参数 |
| --- | --- | --- |
| 机器学习带隙误差修正 | GNN-HSE06 (MACE) | `--backend gnn-hse06` |
| GPU 带隙计算 | ABACUS-HSE06 | `--backend abacus-hse06` |
| GPU 带隙计算 | ABACUS-PBE0 (25% exact exchange) | `--backend abacus-pbe0` |
| 数据驱动泛函带隙计算 | CIDER-PBE0 | `--backend cider-pbe0` |

**特别注意**：“机器学习带隙误差修正”为软件的业务命名，源码的实际实现是
legacy/provisional MACE GNN 带隙预测及直接/间接分类头，不是“输入某个 DFT
计算结果后对误差做算术校正”的独立流程。

HSE06、PBE0、CIDER 是真实不同计算算法。**选择下拉项会改变发往 `bgp predict`
的实际 `--backend` 参数**，不是假按钮。它们不是开箱即用的 GNN 替代品：
ABACUS-HSE06 / ABACUS-PBE0 / CIDER-PBE0 三种计算均需完成专用 GPU/MPI
`abacus_pw_gpu` 构建、赝势/模型资产验证，按要求填写 **ABACUS 可执行文件的 Linux 绝对路径**，
并通过 `bgp doctor --abacus /path/to/abacus_pw_gpu` 核查。
在环境不可用时，界面明确显示不可用；**不会伪造结果**。

### 3. 计算报告在当前工作区显示

选择历史计算任务后点击工具栏 **「计算报告」**，或者点击中央工作区
**「计算报告」** 标签，直接在当前软件窗口显示排版完整的报告。
**不会打开新浏览器标签页，也不会直接弹出打印窗口**。报告按
`result.json` + 当前任务真实数据生成，包含材料、业务类型、实际后端、带隙、
分类、溯源、模型 SHA256、科学参数、警告等；失败的任务显示失败信息，不杜撰带隙。

报告页上可选：
- **导出报告 TXT**：保存当前报告内容到本机；
- **打印 / 保存 PDF**：只有在你主动点击时才调用系统打印对话框；
  不在浏览器中打开独立的报告网页。

### 安全替换与启动

1. 在运行旧版 GUI 的 WSL 终端按 `Ctrl+C` 关闭旧服务（避免 8780 端口占用）。
2. 备份旧版界面目录，然后将本 ZIP 解压至
   `D:\program\gnnbgp-industrial-ui`（保持压缩包内同名顶层文件夹）。
3. **不覆盖或重新安装** `D:\program\gnnbgp`、`.venv/`、`~/BandGap-Studio/workspace`。
4. WSL 启动：

   ```bash
   bash /mnt/d/program/gnnbgp-industrial-ui/scripts/start_wsl.sh
   ```

   如果端口已占用，可使用：

   ```bash
   BGP_UI_PORT=8781 bash /mnt/d/program/gnnbgp-industrial-ui/scripts/start_wsl.sh
   ```

5. 用浏览器访问终端打印的 `http://127.0.0.1:8780` 或 `:8781`，
   按 `Ctrl+F5` 强制刷新静态页面缓存。

### 版本验证与回归测试

- Linux/Python/JavaScript 语法与界面 HTTP 接口模拟回归测试
  位于 `tests/`；可用 `python -m unittest discover -s tests -v` 执行。
- `tests/fake_bgp.sh` **仅供 UI/HTTP/进程调度的模拟测试**，
  其中数值绝不能用于 GNN/DFT 科学准确性认定。
- 必须分别在用户的原生源码环境内核实三种计算类型是否实际可用；
  此项目不编译 ABACUS，也不自带它的可执行文件。

---


## v0.4 彩色拟物图标升级（仅 UI 美化）

这次基于你认可的 v0.3 工业软件布局继续完善，新增至 **52 个彩色 SVG 图标**；**不改变面板结构、按键事件、计算队列、Python 后端和科学模型**。

- 文件导入是带折角与蓝色上传标志的「Windows 资源管理器式」文档图标。
- 已导入材料、甲方内置样例按文件类型显示不同图标：`CIF` 蓝、`POSCAR` 青、`VASP` 蓝灰、`XYZ` 紫；
  `JSON` 橙、`CSV` 绿、`LOG` 灰蓝、`PDF` 红。下载列表也自动匹配类型。
- 文件夹采用类似资源管理器的**金黄色立体文件夹**，带有明暗、折边和阴影。
- 运行绿色、停止红色、刷新蓝色、设置金属蓝灰、警告琥珀色、任务/日志/导出等图标均有彩色细节。
- 这是一套自绘、离线、可修改的 Windows Explorer **风格图标**，不是从 Windows 系统库提取的微软原版 ICO；
  因此在不同浏览器、屏幕缩放下外观一致，也不需要外部字体或 CDN。

图标代码位于 `studio/static/index.html` 的 `<svg class="icon-defs">`；动态扩展名识别位于
`studio/static/app.js` 的 `fileIconFor()`；样式只增补于 `studio/static/style.css`。
`ICONS.md` 包含后续替换、增删图标的指导。

### 从 v0.3 安全升级

1. 在旧版启动窗口按 **Ctrl+C** 停止界面服务。
2. 将这个完整 ZIP 解压覆盖到 `D:\\program\\gnnbgp-industrial-ui`。
3. 在 WSL 中执行：

   ```bash
   bash /mnt/d/program/gnnbgp-industrial-ui/scripts/start_wsl.sh
   ```

4. 浏览器访问 `http://127.0.0.1:8780`，如果还是旧图标，请使用 **Ctrl+F5** 强制刷新。

这次只更换界面文件，不需要重新安装/升级 Python 3.11、PyTorch、MACE，
也无需重新安装甲方 `D:\\program\\gnnbgp`。
已有工作区数据 `~/BandGap-Studio/workspace` **不会被覆盖**。

## v0.3 图标增强（不改甲方计算代码）

本次按照工业软件的功能语义为按钮和标签配备 **45 个本地矢量 SVG 图标**：
- 文件导入：文件向上箭头；甲方样例：文件夹；
- 启动计算：运行三角形；停止：方形停止键；重新运行：环形箭头；
- 刷新、报告、环境诊断：刷新、文件报告、齿轮；
- 结构/结果/任务/比较/输出文件：原子、报告、任务清单、图表、文件组；
- 工程树、计算属性、模型校验、日志、队列、警告、导出、重命名等都有对应图标。

**全部图标内置在 `studio/static/index.html`，无需联网、无需安装图标字体。**
菜单中动态图标映射在 `studio/static/app.js` 的 `menuIconNames`，
统一大小/颜色在 `studio/static/style.css` 的 `.ui-icon` 等规则中。

### 从 v0.2 升级

1. 如果旧版窗口仍在运行，回到启动它的 WSL 终端按 `Ctrl+C` 停止。
2. 备份旧版界面源码（可选），把本 ZIP 解压覆盖至
   `D:\program\gnnbgp-industrial-ui`（保持压缩包中的文件夹结构）。
3. 继续运行 `bash /mnt/d/program/gnnbgp-industrial-ui/scripts/start_wsl.sh`。
4. 浏览器打开 `http://127.0.0.1:8780`。若图标显示旧样式，
   按 `Ctrl+F5` 强制刷新缓存。

**不需要重新安装 Python/torch/MACE；本版只替换 GUI 项目文件。**
你的甲方 `D:\program\gnnbgp` 原始源码不被修改，
已有计算数据位于 `~/BandGap-Studio/workspace`，不在本 ZIP 内。


## 适配的是什么代码？

**唯一适配甲方新版源码**：Windows `D:\program\gnnbgp`，
WSL Ubuntu `/mnt/d/program/gnnbgp`。不再使用旧的
`bandgap-predictor-0.1.0-handoff.tar.gz`、旧仓库以及旧版 Web 界面代码。

适配关键点已按本次提供的 `gnnbgp.zip` 源码确认：

- `src/bandgap_predictor/cli.py`：CLI 入口 `bgp`，提供 `predict`、`backends`、
  `assets verify`、`doctor`；其中 `predict` 接收 `STRUCTURE --backend X
  --work-root DIR [--abacus EXE]`。
- `src/bandgap_predictor/schemas.py`：`PredictionResult`、
  `PredictionValue`、`BackendId` 的字段定义。
- `src/bandgap_predictor/runs.py`：每次计算有唯一子目录，末尾写入
  `result.json`，保留失败和中断记录。
- `src/bandgap_predictor/structures.py`：必须是一个三维周期性、无磁性初始化
  的晶体结构；GUI 上传时也按这些约束预检。
- `src/bandgap_predictor/backends/gnn.py`：GNN 输出数值带隙，
  并可能提供 `gap_type` / `gap_type_probability`；模型按新源码资产验证。
- `src/gnnbgp/`、`assets/models/gnn/`：实际模型和特征实现，
  **界面不导入或修改这些私有科学实现**。
- `vendor/abacus/`、`tools/build_abacus.py`：三种第一性原理计算需要单独编译
  并通过 `bgp doctor`，不靠 GUI 本身安装。

这个 GUI 不调用尚未实现的 `bgp bands`、`bgp serve`、`--json` 标志。
不伪造 E–k 能带，也不提供锁定科学预设以外的自定义计算参数。

## 工业风格界面功能

| 工作区 | 功能 |
|:--|:--|
| 菜单与工具栏 | 导入、示例、计算/终止/重跑、刷新、报告、环境诊断、帮助、快捷键 |
| 工程树 | 本地持久化材料库、搜索筛选、样例导入、重命名、移除未使用材料 |
| 结构编辑区 | 交互式三维晶胞投影、原子坐标表、晶格参数、元素标签、坐标轴 |
| 属性面板 | 材料属性、四种后端、GNN CPU/CUDA、ABACUS 路径、任务备注、锁定预设 |
| 任务管理 | 单任务队列、任务列表筛选、状态、取消、重复运行、异常状态 |
| 结果分析 | 带隙 eV、电子态、直接/间接、类型预测概率、SCF、耗时、科学方法 |
| 可复现性 | `result.json`、设置、模型哈希、来源、占据态、警告、日志及文件目录 |
| 数据比较 | 基于成功运行的结果对比条形图、CSV 导出 |
| 工作底栏 | 实时日志、队列、错误警告；可拖动高度 |
| 工业工作台 | 布局分隔栏可拖动，功能标签页，状态栏，本地打印计算报告 |

前端为纯 HTML/CSS/JS，不依赖 CDN；Python 服务使用标准库，
晶体校验使用已安装的 ASE。

## 用户当前机器的最短安装与启动

**你已在本机验证现有 GNN 命令成功**，无需重新安装 Python、PyTorch、MACE、
e3nn，也不应运行旧版安装脚本。

将本压缩包解压到 Windows 目录：

`D:\program\gnnbgp-industrial-ui`

在 WSL Ubuntu 中，启动：

```bash
bash /mnt/d/program/gnnbgp-industrial-ui/scripts/start_wsl.sh
```

浏览器打开：

**http://127.0.0.1:8780**

默认源代码读取 `/mnt/d/program/gnnbgp`；
环境使用 `/home/guojin/BandGap-Studio/.venv`；
工作区保存至 `/home/guojin/BandGap-Studio/workspace`。

本软件只在 `127.0.0.1` 监听，不需要账户绑定、不依赖互联网，
并且不会修改甲方 `src/`、`assets/`、`vendor/`。

如果压缩包解压到了其他地方，按实际路径执行其中的 `scripts/start_wsl.sh`。

### 启动前可选检查

```bash
bash /mnt/d/program/gnnbgp-industrial-ui/scripts/check_environment.sh
```

它执行你已跑通的 `pip check`、`bgp backends`、`bgp assets verify`，
不会修改系统和依赖。

### 全新机器安装（当前机器不需要）

如果以后在新的 Ubuntu 上安装、且不存在任何 `.venv`，可执行：

```bash
bash /mnt/d/program/gnnbgp-industrial-ui/scripts/setup_wsl.sh
```

该脚本按甲方 `requirements.txt` 固定 PyTorch 2.2.1 和
MACE 0.3.14，再 `pip install -e /mnt/d/program/gnnbgp` 安装基础依赖
（包括 `platformdirs`）。它不会安装冲突的 `[gnn]` extra 或按
`uv.lock` 换成另一套 PyTorch 2.10.0 / MACE 0.3.9。
检测到已有 `.venv` 会直接停止，不覆盖你的实测环境。

## 第一次使用

1. 点击左侧 **甲方测试样例 → InSb.cif**。这会将样例复制到软件工作区，
   不更改甲方源文件。
2. 右侧方法选 **GNN-HSE06**，设备选 **CPU**。
3. 点击工具栏或属性面板的 **启动计算**。
4. 在底部日志观察真正的 `bgp predict` 进度。
5. 运行结束后点击 **计算结果**，看到真实的带隙、直接/间接类型，
   原始模型警告与 SHA256；在 **输出文件** 下载 `result.json` 与日志。

你之前从这台 WSL 实测得到：

```text
InSb.cif / gnn-hse06
Band gap: 0.2487309 eV
Gap type: direct
```

甲方提供的另一参考值为 `0.2495250 eV / direct`。
这两个结果**不完全一致**，界面会显示你本机真实结果，不会改写成
甲方的参考值。是否满足数值回归需要甲方明确容差和比较设置。

## 常见问题

**Q：计算数值是不是演示数据？**

不是。正式 GUI 始终通过
`/home/guojin/BandGap-Studio/.venv/bin/bgp predict`
启动真实子进程，解析计算产物 `result.json`。
`tests/test_server.py` 使用的 stub 仅用于自动化验证工作流。

**Q：能看到电子能带 E–k 图吗？**

不行。新源码当前版本没有可调用的真实 band-path 结果命令。
提供预测带隙数值、电子态类别、直接/间接类型及原始诊断，不造假。

**Q：四种后端都可用吗？**

不一定。你本机 GNN 已实测成功，其余三种需要专门的
`abacus_pw_gpu` 及匹配的 `.bgp-build.json` 清单，并通过
`bgp doctor --abacus /path/to/abacus_pw_gpu`。

**Q：保存了哪些文件？**

工作区在 `~/BandGap-Studio/workspace`，包含：
- `studio.sqlite3`：材料索引与历史任务；
- `materials/<id>/...`：用户上传和样例的独立副本；
- `jobs/<id>/input/`：本次任务使用的原始结构副本；
- `jobs/<id>/console.log`：GUI 层捕获的标准输出与错误；
- `jobs/<id>/engine-runs/`：由甲方程序生成的目录（含 `result.json` 等）。

**Q：能部署到公网吗？**

不能。当前是单机 WSL GUI 服务，无登录、角色和远程权限隔离，
只监听本机 `127.0.0.1:8780`。不要改变绑定地址对外公开。

**Q：停止进程会丢掉日志吗？**

GUI 给真实 `bgp` 进程组发送 SIGINT，甲方程序尽可能保存
`interrupted` 的 JSON 和日志。GUI 记录也会持久化。
如果整个 GUI 服务异常崩溃，下次启动将未完成的记录标记为
“中断/状态不明”，并提醒检查残留系统进程。

## 文件说明

```text
gnnbgp-industrial-ui/
├── README.md                     中文安装、功能与适配说明
├── studio/
│   ├── server.py                  WSL 本地 REST/任务/SQLite/CLI 服务
│   └── static/
│       ├── index.html             工业工作台窗口结构
│       ├── style.css              工业桌面风格
│       └── app.js                 前端交互、视图、结构投影
├── scripts/
│   ├── start_wsl.sh               复用你已跑通的环境启动
│   ├── check_environment.sh       只读诊断
│   └── setup_wsl.sh               全新机器环境安装（非当前机器）
└── tests/
    └── test_server.py             模拟 CLI 的后端集成测试
```

## 验收边界

已测试：Python/JavaScript 语法、独立文件结构、后端模拟 CLI 完整流程、
日志与 JSON 持久化、下载路径约束及服务 HTTP 返回（参见测试结果）。

**尚未**在你的 WSL 系统中运行这份 GUI 或完成实际鼠标交互测试；
真实 GNN 本机已由你自己运行成功，GUI 从浏览器启动的真实计算仍需要现场验收。

本界面不对 GNN 模型精度、ABACUS 编译、SOC/磁性支持做未经证实的保证。
