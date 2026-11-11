# GNNBGP Industrial Workbench v0.6

基于用户上传的 `gnnbgp-industrial-ui.zip`（v0.5）增量修改。

## 修改
- `scripts/start_wsl.sh`：自动识别 `BGP_ABACUS`，添加 `BGP_DEPS_PREFIX/lib`
  到 `LD_LIBRARY_PATH`，保留已有动态库路径；避免把任务数据放到 `/mnt/<盘符>/`。
- `studio/server.py`：为功能 2/3 自动指定已编译的 ABACUS 绝对路径，
  提交任务前验证文件可执行；功能 1 不依赖 ABACUS；运行日志记录
  真实 backend、Linux work-root 和 ABACUS 路径；系统接口返回环境信息。
- `studio/static/app.js`：GPU/CIDER 方法自动填入启动脚本识别的 ABACUS 路径，
  保留用户手工配置；根据真实可执行文件状态给出操作提示。
- `studio/static/index.html`：更新 ABACUS 路径说明和版本号。
- 测试：补充默认 ABACUS 路由、缺失二进制处理、XYZ 上传三后端测试。
- 新增《GUI_三算法联调指南_v0.6.md》。

## 沿用的 v0.5 功能
三类业务下拉菜单、真实 CLI backend 路由、XYZ/CIF/VASP/POSCAR 导入、
任务进度、原生 JSON 结果、工作区内计算报告、删除材料保留任务历史。

## 限制
科学引擎/ABACUS 仍在用户自己的 WSL 中执行；本 ZIP 不包含
甲方算法源码、GPU 可执行程序、模型权重、赝势或任何 DFT 模拟结果。
测试 `fake_bgp.sh` 仅用于 UI/HTTP/进程调度回归。
