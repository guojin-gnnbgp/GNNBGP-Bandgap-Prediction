# GNNBGP v0.6 · 三算法与工业 GUI 联调指南

本指南适用于：Windows + WSL Ubuntu 24.04、RTX 5060 Ti、已有 `~/BandGap-Studio/.venv`、
已编译安装的 GPU ABACUS。本 GUI **不包含 ABACUS 二进制、不修改 DFT 参数、不重装机器学习环境**。

## 一、文件与运行环境

| 项目 | Linux 路径 |
|---|---|
| GUI 代码 | `/mnt/d/program/gnnbgp-industrial-ui` |
| 算法源码 | `/mnt/d/program/gnnbgp` |
| bgp CLI | `/home/guojin/BandGap-Studio/.venv/bin/bgp` |
| ABACUS GPU | `/home/guojin/BandGap-Studio/build/abacus-install/bin/abacus_pw_gpu` |
| libxc/FFTW 依赖 | `/home/guojin/bgp-deps/lib` |
| GUI 任务/数据 | `/home/guojin/BandGap-Studio/workspace` |

应保持 GUI 计算数据在 `/home/guojin/` 等区分大小写的 Linux 文件系统中。
**不要设置 `BGP_UI_DATA=/mnt/d/...`**。ABACUS 需要名为 `INPUT` 的普通文件，
引擎同时保存小写 `input/` 目录，Windows 挂载盘可能产生名称冲突。

## 二、启动前只读检查（无需重装）

在 WSL 中执行：

```bash
test -x "$HOME/BandGap-Studio/.venv/bin/bgp" && echo "bgp OK"
test -x "$HOME/BandGap-Studio/build/abacus-install/bin/abacus_pw_gpu" && echo "ABACUS OK"
test -d "$HOME/bgp-deps/lib" && echo "deps OK"

"$HOME/BandGap-Studio/.venv/bin/bgp" doctor \
  --abacus "$HOME/BandGap-Studio/build/abacus-install/bin/abacus_pw_gpu"
```

使用 `bgp doctor` 检查构建清单和特性；它不能代替真实计算验收。

## 三、部署本 GUI v0.6

1. 在现有 GUI 终端按 `Ctrl+C` 关闭旧版服务。
2. **先备份** `D:\program\gnnbgp-industrial-ui`，再将新的
   `gnnbgp-industrial-ui` 文件夹内容覆盖该 GUI 目录。
3. 不要覆盖 `D:\program\gnnbgp`、`~/BandGap-Studio/.venv`、
   `~/BandGap-Studio/build/abacus-install` 或 `~/BandGap-Studio/workspace`。
4. 在 WSL 执行：

```bash
bash /mnt/d/program/gnnbgp-industrial-ui/scripts/start_wsl.sh
```

默认浏览器地址：`http://127.0.0.1:8780`。如需改变端口：

```bash
BGP_UI_PORT=8781 bash /mnt/d/program/gnnbgp-industrial-ui/scripts/start_wsl.sh
```

启动脚本只负责设置路径和环境变量，**不会安装、升级或重新编译任何科学软件**。
默认自动设置 `BGP_ABACUS`、`BGP_DEPS_PREFIX`，并向
`LD_LIBRARY_PATH` 前置 `bgp-deps/lib`，保留已有其他库路径。
如实际安装位置不同，启动前设置：

```bash
export BGP_ABACUS="/你实际安装的/abacus_pw_gpu"
export BGP_DEPS_PREFIX="/你的/bgp-deps"
bash /mnt/d/program/gnnbgp-industrial-ui/scripts/start_wsl.sh
```

## 四、联调顺序和操作

1. 在浏览器进入工业工作台，导入你已经验证过可运行的 `InSb.xyz`。
   不建议为本次联调改变 XYZ 内容或现有计算参数。
2. 先在右侧选择 **机器学习带隙误差修正**，
   方法 `GNN-HSE06`，点击“提交计算任务”。
3. 在“任务管理”查看状态，“输出”查看真实 CLI 日志，
   在“计算结果”和“计算报告”查看结果。
4. 再选择 **数据驱动泛函带隙计算**，方法 `CIDER-PBE0`。
   ABACUS 路径应已自动填入；可打开“工具 → ABACUS 诊断”。
5. 最后选择 **GPU 带隙计算**，方法 `ABACUS-HSE06`，
   使用同一份已确认的 GPU ABACUS 程序。
6. 每个任务都应有独立日志、原始结构副本和 `result.json`。
   不应把“进程已启动”当作“计算成功”。

| 业务类型 | CLI backend | 引擎 |
|---|---|---|
| 机器学习带隙误差修正 | `gnn-hse06` | MACE/GNN |
| GPU 带隙计算 | `abacus-hse06` | GPU ABACUS + HSE06 |
| 数据驱动泛函带隙计算 | `cider-pbe0` | GPU ABACUS + CIDER-PBE0 |

GUI 还保留 `abacus-pbe0` 作为“GPU 带隙计算”的另一种可选方法，
但它不在这次三项已跑通算法的必选验收清单中。

## 五、任务与结果文件

GUI 数据库：`~/BandGap-Studio/workspace/studio.sqlite3`。

每个任务均独立归档：

```text
~/BandGap-Studio/workspace/jobs/<任务ID>/
  input/<导入的结构.xyz>
  console.log
  engine-runs/<bgp产生的计算目录>/
    INPUT
    KPT
    ...
    result.json
```

实际计算目录名称由 `bgp` 引擎生成；成功、失败或取消的任务都有
状态信息。若引擎未生成有效结果，GUI 会记录失败和日志，不显示虚构的带隙。

如果启动时没有检测到 ABACUS 路径，功能 1 仍然可运行，
功能 2/3 的提交将给出可执行文件校验错误。可手动填写
Linux 绝对路径并在“环境诊断”中运行 doctor。

## 六、常见问题

- **`IsADirectoryError .../INPUT`**：任务工作目录误放在 Windows 挂载盘。
  检查启动输出 `Workspace` 是否位于 `/home/guojin/...`。
- **`ABACUS 不存在或不可执行`**：检查 GUI 右侧自动填入的路径，
  使用 `ls -lh` 及 `test -x` 检查文件。
- **`libxc`、CUDA、MPI 共享库缺失**：在启动 GUI 的 WSL 终端检查
  `echo "$LD_LIBRARY_PATH"`，确认依赖路径；不建议擅自替换已调通的编译工具链。
- **任务长时间运行**：HSE06 EXX 阶段可能耗时很长。
  通过任务日志、`nvidia-smi`、`pgrep -af abacus_pw_gpu` 辅助判断。
- **浏览器没有显示更新**：先关闭旧 GUI 服务，重启 v0.6，
  然后用 `Ctrl+F5` 刷新浏览器。
- **GNN 可运行但 ABACUS 不可用**：三种方法依赖不同运行时；
  不要将 GNN 成功视为 HSE06 科学验证通过。

## 七、验收原则

须在用户自己的 WSL + RTX 5060 Ti 环境中实际提交三类任务，
逐项确认后台方法、任务目录、返回码、`result.json`、
带隙数值及报告一致。**本 ZIP 的自动化测试只用模拟 CLI 校验
GUI/HTTP/任务路由，不表示已经在该机器上完成真实 GPU 回归。**
