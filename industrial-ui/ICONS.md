# GNNBGP Industrial Workbench — 图标维护说明

# v0.4 — 彩色 Windows 资源管理器风格

图标现在是带有渐变、折角、阴影、文件扩展名标签的彩色内嵌 SVG。
图标依旧自包含，不依赖 Windows `.ico`、版权字体或互联网。
如需与某一特定 Windows 11 版本的原生图标像素级一致，需另行获得合法的
Windows 资源图标资产，并替换当前自绘 SVG；当前版本只承诺风格接近。

文件类型图标命名：`file-cif`、`file-poscar`、`file-vasp`、`file-xyz`、
`file-json`、`file-csv`、`file-log`、`file-pdf`、`file-text`。

要调整配色：编辑 `index.html` 内 SVG 根节点的线性渐变颜色，如
`bgp-blue`、`bgp-folder`、`bgp-paper`、`bgp-green`、`bgp-red`。
要调整文件类型与图标关联：编辑 `app.js` 的 `fileIconFor()`。
**不要更改**甲方计算 CLI、JSON Schema 和科学参数预设。


本版不依赖 FontAwesome、在线 CDN 或系统 Emoji。所有按钮和标签使用相同线宽的矢量图形，
适合工业软件高密度工具栏和高 DPI 显示器。

## 图标位于哪里？

- `studio/static/index.html`：最前面的 `<svg class="icon-defs">` 定义 45 个 `<symbol>`；
- 同文件里的工具栏和标签按钮通过 `<svg><use href="#i-file-up"></use></svg>` 引用；
- `studio/static/app.js`：动态项目树 `iconHTML()`；菜单名与图标对应由 `menuIconNames` 控制；
- `studio/static/style.css`：`.ui-icon`、`.tab-icon`、`.folder-icon`、`.menu-item-leading` 等尺寸、间距、颜色。

## 增加新图标的步骤

1. 在 `index.html` 的 `icon-defs` 内新增一个 `<symbol id="i-name" viewBox="0 0 24 24">`。
2. 继续用 24×24 的线性图标路径 (`fill="none"`、`stroke="currentColor"`、线宽 1.7)。
3. 在静态按钮中使用 `<svg class="ui-icon" aria-hidden="true"><use href="#i-name"></use></svg>`。
4. 如果是 JS 动态菜单，在 `menuIconNames` 映射中使用 `"菜单文本":"name"`。
5. 检查按钮始终有文字或 `aria-label` / `title`，图标本身 `aria-hidden="true"`。

## 不应改动什么？

图标是 GUI 元素，不影响甲方的 `bgp predict` 计算命令与锁定科学预设。
不要借新增图标制造不存在的 E–k 能带计算按钮、假计算数据或未实现的算法入口。
