# 文档配图流程

当用户要求为项目文档画图、流程图、架构图、规则图或 draw.io 图时，按此流程处理。

## 默认判断

1. 先读取用户指定的文档、章节或代码。
2. 如果只是简单流程，优先使用 Mermaid，直接写入 Markdown / MDC。
3. 如果图较复杂、需要可视化编辑、适合作为长期资产，使用 drawio MCP。
4. 中文文档默认使用中文节点文字。

## draw.io 资产约定

- `.drawio` 源文件保存到 `docs/diagrams/`。
- 先修改和确认 `.drawio` 原图；原图确认无误后，最后再导出同名 `.svg`。
- 不手写导出的 `.svg` / `.png`，导出文件只作为展示产物。
- Markdown / MDC 中使用相对路径引用图片。

示例：

```text
docs/diagrams/perpetual-chase.drawio
docs/diagrams/perpetual-chase.svg
```

```md
![长捉判定流程](./diagrams/perpetual-chase.svg)
```

## 使用 drawio MCP 时

1. 启动 drawio MCP 会话。
2. 根据文档内容生成 draw.io 图。
3. 只保存或更新 `.drawio` 原图。
4. 等用户确认原图后，再导出 `.svg` 并插入引用。

## 画图要求

- 节点名称短而明确。
- 判断节点使用问题句。
- 流程顺序必须和规则或代码执行顺序一致。
- 不要为了好看改变规则含义。
