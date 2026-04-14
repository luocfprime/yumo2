# 使用说明

## 启动

```bash
yumo2 gui --data path/to/field.plt --mesh path/to/mesh.stl
```

如果在 Ubuntu 的 WSL 中启动后立即 segfault，可改用软件 OpenGL：

```bash
LIBGL_ALWAYS_SOFTWARE=1 yumo2 gui --data path/to/field.plt --mesh path/to/mesh.stl
```

如果连 GUI 转发都不可用，请先按安装文档中的 WSL 依赖说明安装 X11 / OpenGL 相关包。

## 网格表面

标量场通过UV展开（xatlas）和三线性插值烘焙到网格上。UV展开较慢，每次启动时对每个网格执行一次。

### Colormap与颜色范围

从内置列表中选择colormap，调整颜色最小/最大值，或重置为数据的完整范围。

### 标量变换

在 **Processing** 面板中选择变换方式：`identity`（默认）、`log_e` 或 `log_10`。变换作用于表面采样值，非正值替换为变换后的最小值。

### 去噪

启用高斯去噪并调整 sigma 值（单位：texel）对烘焙纹理进行平滑。

## Scope（范围查询）

在球形范围内查找最小或最大标量值。在 **Scope** 面板中启用，设置半径和查询模式，然后在视口中点击即可查询。

## 坐标拾取器（Coord Picker）

点击网格表面读取该点的标量值。展开 **Coord Picker** 面板即可激活。

## 截图（Snapshot）

将当前视口导出为带有嵌入颜色条的PNG图像。在 **Snapshot** 面板中调整裁剪区域和颜色条位置/尺寸，然后点击 **Save PNG**。

!!! tip "Live Preview 与性能"

    取消勾选 **Live Preview** 可冻结预览，降低CPU占用，从而在交互操作时获得更高、更流畅的帧率。

## 配置（Profiles）

所有设置均以命名配置的形式保存在 `.yumo2/profiles/` 目录中。使用 **Profile** 面板创建、重命名或切换配置。
