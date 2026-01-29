## 配准可视化工具（Python Desktop GUI）

这是一个独立的 Python 桌面应用，用于：
- 通过配置运行多种外部配准算法（不局限于 `exe`，也支持 MATLAB / PyTorch 等）
- 选择输入（文件夹模式 / 图片对模式）
- 读取或生成 `matches.txt`（x1 y1 x2 y2）
- 用 OpenCV RANSAC 估计仿射变换（输出 3x3 矩阵）替代 FSC
- 生成两种可视化：匹配点可视化、棋盘融合可视化
- 在界面中展示两张结果图、运行日志与变换矩阵

### 目录
- `app/` 主程序代码
- `algorithms/` 推荐的算法工程放置目录（exe / matlab / pytorch）
- `user_config.json` 运行配置（算法名称与启动命令、默认参数等）
- `outputs/` 默认输出目录（运行后生成）

## 运行

1) 创建虚拟环境（推荐）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) 安装依赖

```powershell
pip install -r requirements.txt
```

3) 启动

```powershell
python -m app
```

**注意**：如果遇到 `ImportError: DLL load failed` 或 PySide6 相关错误，程序会自动降级使用 **Tkinter** 界面（样式较简单，但功能一致）。
你也可以强制使用 Tkinter 界面：
```powershell
python -m app --tk
```

## 运行测试

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## 输入说明

### 图片对模式
- Fixed：目标图（作为对齐参考）
- Moving：待对齐图（将被变换/融合展示）

### 文件夹模式
会自动扫描形如：
- `*_1.jpg` 与 `*_2.jpg`
- `*_1.png` 与 `*_2.png`

同名（去掉 `_1/_2` 后）会组成一个 pair，列表中可选择后运行。

## 输出说明

默认输出到：`python_registration_gui/outputs/<算法名>/<pair名>/`
- `matches.txt`：匹配点（x1 y1 x2 y2）
- `H_affine_3x3.txt`：估计的 3x3 变换矩阵
- `matches_vis.jpg`：匹配点可视化
- `checkerboard.jpg`：棋盘融合可视化
- `warped.jpg`：变换后的 Moving 图（用于 Compare 对比）
- `run.log`：运行日志

## 算法接入（配置驱动）

GUI 通过 `user_config.json` 的 `algorithms` 列表来展示可选算法，每个算法由：
- `name`：界面显示名称
- `command`：启动命令模板（支持占位符）
- `cwd`：运行工作目录（可留空）

占位符：
- `{fixed}` / `{moving}`：输入图像路径
- `{matches_out}`：算法需要写出的 `matches.txt` 路径
- `{out_dir}`：本次任务输出目录
- `{repo_root}`：python_registration_gui 根目录
- `{algorithms_root}`：`algorithms/` 目录绝对路径

推荐把后续算法项目放到 `python_registration_gui/algorithms/` 下，详细约定见 [README.md](file:///d:/hand_craft_registration/SRIF-master/python_registration_gui/algorithms/README.md)。

## 大文件托管

权重/模型/数据集/大型 exe 的推荐放置与托管方式见 [LARGE_FILES.md](file:///d:/hand_craft_registration/SRIF-master/python_registration_gui/LARGE_FILES.md)。
