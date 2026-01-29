## 大文件怎么放（权重 / 模型 / EXE / 数据集）

这个仓库建议保持“源码 + 配置”为主，尽量不要直接提交大文件（几十 MB 以上的 exe / ckpt / 数据集等），否则会导致：
- clone 很慢
- 版本膨胀严重
- 同事更新成本高

### 推荐做法（按场景选）

#### 1) 放在本仓库目录内，但不提交（最简单）

适用于：算法 EXE、模型权重、临时输出等。

- 放置位置（推荐）：
  - `python_registration_gui/algorithms/exe/`：exe
  - `python_registration_gui/algorithms/**/weights/`：权重
- 这些目录已被 `.gitignore` 忽略，不会进 Git。
- 你只需要把 `user_config.json` 里的 `command` 写成使用 `{algorithms_root}` 的相对引用即可。

缺点：你发仓库给别人时，需要额外把这些大文件一起打包发给对方或让对方下载。

#### 2) Git LFS（想“跟着仓库走”但文件很大）

适用于：确实希望权重/二进制也版本化管理的团队。

大致步骤（示意）：
- 安装 git-lfs
- `git lfs track "*.ckpt" "*.pth" "*.pt" "*.onnx" "*.exe"`
- 提交 `.gitattributes`

优点：文件跟着仓库版本走；缺点：需要 LFS 服务配额/带宽（GitHub 有限制）。

#### 3) 外部托管（最推荐给“发给别人用”的场景）

适用于：模型权重、数据集、超大 exe。

常见选择：
- GitHub Releases（适合打包的 exe/权重作为“发布资产”）
- HuggingFace Hub（非常适合 ckpt/pt/pth/onnx）
- 阿里云 OSS / 腾讯 COS / S3 / 公司的 NAS / 网盘
- 数据集：DVC + Remote（需要可复现数据版本管理时）

建议做法：
- 在 `algorithms/**/weights/` 里只放一个 `README` 或 `download.ps1` / `download.sh`（脚本可以从外部下载权重）
- `user_config.json` 里仍然引用 `{algorithms_root}` 下的本地路径，第一次运行前先下载到对应位置即可

### 目录约定

GUI 的算法统一从 `python_registration_gui/algorithms/` 下找：
- `algorithms_root` 允许留空，程序会自动回退到默认 `.../python_registration_gui/algorithms`

`user_config.json` 的 `command` 支持占位符：
- `{fixed}` `{moving}` `{matches_out}` `{out_dir}` `{repo_root}` `{algorithms_root}`

