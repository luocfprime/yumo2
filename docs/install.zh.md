# 安装

**前提条件：** 需要安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

从PyPI安装：

```bash
uv tool install yumo2
```

或者不安装直接运行一次：

```bash
uv run --with yumo2 yumo2 gui --data field.plt --mesh mesh.stl
```

## Ubuntu on WSL

先安装常用的 X11 / OpenGL 运行库：

```bash
sudo apt update
sudo apt install -y libglu1-mesa mesa-utils x11-apps
```

先确认图形转发可用：

```bash
echo $DISPLAY
xeyes
```

如果 GUI 打开黑窗后立刻发生 segfault，请使用软件 OpenGL 运行：

```bash
uv tool install yumo2
LIBGL_ALWAYS_SOFTWARE=1 yumo2 gui --data field.plt --mesh mesh.stl
```

或者不安装直接运行一次：

```bash
LIBGL_ALWAYS_SOFTWARE=1 uv run --with yumo2 yumo2 gui --data field.plt --mesh mesh.stl
```

其中 `mesa-utils` 可用于 `glxinfo` 之类的诊断命令，`x11-apps` 可用于快速确认 X11/WSLg 图形转发是否正常。
