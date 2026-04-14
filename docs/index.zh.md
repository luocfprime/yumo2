# yumo2

3D标量场网格表面可视化工具。

<video width=100% autoplay muted loop>
  <source src="[[url.videos]]/media/movies/teaser.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

yumo2将体积标量场通过UV展开和三线性插值映射到三角网格表面，并使用 [Polyscope](https://polyscope.run/) 进行交互式渲染。

## 功能

- 支持颜色范围调整的colormap选择
- 原始标量值的对数/线性处理
- 烘焙纹理的高斯去噪
- 交互式Scope工具：在指定半径内查找最小/最大值
- 截图导出，带有嵌入的颜色条
