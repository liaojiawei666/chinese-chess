# 文档配图流程

当用户要求为项目文档画图、流程图、架构图、规则图或 draw.io 图时，按此流程处理。

本流程**不依赖任何 MCP 服务**：AI 直接生成 `.drawio`（mxGraphModel XML）源文件，再用本机的 draw.io Desktop 命令行导出 `.svg`。任何支持读取本仓库文件、能写文件并执行命令的 AI 工具都可遵循本流程。

## 默认判断

1. 先读取用户指定的文档、章节或代码。
2. 如果只是简单流程，优先使用 Mermaid，直接写入 Markdown / MDC。
3. 如果图较复杂、需要可视化编辑、适合作为长期资产，生成 `.drawio` 源文件。
4. 中文文档默认使用中文节点文字。

## draw.io 资产约定

- **所有图文件（含临时图、测试图）都必须放在项目目录内的 `docs/diagrams/`，绝不放到 `/tmp` 或项目以外的任何地方。**
- 先生成并确认 `.drawio` 原图；**必须等用户明确确认后**，再导出 `.svg` 并写入 Markdown。未确认前不导出、不改任何文档。
- 不手写导出的 `.svg` / `.png`，导出文件只作为展示产物。
- Markdown / MDC 中使用相对路径引用图片。
- 文件名用小写加连字符，名称要能反映图的内容。

示例：

```text
docs/diagrams/perpetual-chase.drawio
docs/diagrams/perpetual-chase.svg
```

```md
![长捉判定流程](./diagrams/perpetual-chase.svg)
```

## 生成与导出步骤

严格按顺序执行，**第 3、4 步在用户确认前不得进行**：

1. 生成 draw.io XML（见下方「XML 格式」），写入 `docs/diagrams/<name>.drawio`（必须在项目目录内）。
2. 把原图展示给用户，**等待用户明确确认**。在用户说确认 / 可以导出之前，停在这一步，不导出、不改文档。
3. 用户确认后，才用 draw.io Desktop CLI 导出同名 `.svg`，输出路径同样在 `docs/diagrams/` 内（保留 `.drawio` 源文件，不删除）：

```bash
/Applications/draw.io.app/Contents/MacOS/draw.io -x -f svg -e -b 10 \
  -o docs/diagrams/<name>.svg docs/diagrams/<name>.drawio
```

   关键参数：
   - `-x` 导出模式
   - `-f svg` 输出格式（也可 `png` / `pdf`）
   - `-e` 把图的 XML 嵌入导出文件，使 `.svg` 仍可在 draw.io 中打开编辑
   - `-b 10` 图四周留 10px 边距
   - `-o` 输出路径

4. 导出成功后，再在用户指定的 Markdown / MDC 位置以相对路径插入 `.svg` 引用。若用户指定了具体章节 / 位置，就插到那里，不要随意放置。

> 注意：与官方 Skill 不同，本仓库**同时保留** `.drawio` 源文件和导出的 `.svg`，方便后续再编辑。因此 `.svg` 用单扩展名（`name.svg`），不用 `name.drawio.svg`。

## XML 格式

`.drawio` 文件就是 mxGraphModel XML，直接生成 XML，不要用 Mermaid / CSV（那需要服务端转换，无法存为原生文件）。

### 基本结构

每张图都必须有这个骨架：

```xml
<mxGraphModel adaptiveColors="auto">
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    <!-- 图元素放这里，parent="1" -->
  </root>
</mxGraphModel>
```

- `id="0"` 是根层
- `id="1"` 是默认父层
- 所有图元素用 `parent="1"`（除非使用多图层）

### 完整 XML 参考

边的路由、容器、图层、tag、元数据、暗色模式配色、样式属性等完整参考，见官方单一来源：

https://raw.githubusercontent.com/jgraph/drawio-mcp/main/shared/xml-reference.md

## XML 良构（务必遵守）

- **绝不**在输出里写任何 XML 注释（`<!-- -->`）—— 既费 token，又可能引发解析错误。
- 属性值里的特殊字符要转义：`&`、`<`、`>`、`"`。
- 每个 `mxCell` 用唯一的 `id`。
- 每条边（edge）的 `mxCell` 必须包含子元素 `<mxGeometry .../>`，不能写成自闭合标签，否则边不渲染。

## 画图要求

- 节点名称短而明确。
- 判断节点使用问题句。
- 流程顺序必须和规则或代码执行顺序一致。
- 不要为了好看改变规则含义。

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| CLI 找不到 | draw.io Desktop 未安装或不在该路径 | 确认 `/Applications/draw.io.app` 存在；保留 `.drawio` 让用户手动打开 |
| 导出文件空白 / 损坏 | XML 不合法（如注释里的双连字符、未转义字符） | 写入前校验 XML 良构 |
| 图打开是空的 | 缺少根层 `id="0"` 和 `id="1"` | 补全基本骨架 |
| 边不渲染 | 边的 `mxCell` 自闭合、缺 `mxGeometry` 子元素 | 每条边补 `<mxGeometry relative="1" as="geometry"/>` |
