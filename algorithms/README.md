## algorithms 目录约定

这个目录用于集中放置所有“可被 GUI 调用”的算法工程（不局限于 EXE，也可以是 MATLAB 工程、PyTorch/Python 工程等）。

### 推荐结构

- `algorithms/exe/`：已打包好的可执行文件（.exe）
- `algorithms/matlab/`：MATLAB 工程（.m/.mlx + 依赖文件）
- `algorithms/pytorch/`：PyTorch 工程（通常是 python 脚本 + 权重）

你也可以在 `algorithms/` 下自定义更多子目录，只要在配置里写对 `command` 和 `cwd` 即可。

### GUI 约定（最重要）

无论算法以什么形式存在，GUI 的流水线都需要算法最终产生一个 `matches.txt`，并且写到 `{matches_out}` 指定的位置。

### 配置中的环境说明

在 `user_config.json` 的每个 `algorithms[]` 条目里可以新增：

- `env_hint`：环境/依赖/使用方式说明，会在 GUI 里随算法选择展示

配置里的 `command` 支持以下占位符：

- `{fixed}`：参考图路径（Fixed / Target）
- `{moving}`：待配准图路径（Moving / Source）
- `{matches_out}`：算法需要写出的匹配文件路径（matches.txt）
- `{out_dir}`：本次任务的输出目录
- `{repo_root}`：python_registration_gui 根目录
- `{algorithms_root}`：本目录（algorithms）的绝对路径

`cwd` 也支持同样的占位符；留空则默认使用 `{repo_root}`。
