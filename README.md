# monitor-space-clean

摄像头录像中的人物片段筛选工具，使用 NVIDIA CUDA、PyTorch、Ultralytics YOLO 和 FFmpeg。

**归档用途：原部署已决定退役，不再需要常驻运行。仓库保留原始脚本，不自动安装或启用后台服务。**

## 重要：默认行为会删除原录像

这不是只生成检测报告的工具。非 `--dry-run` 模式下：

1. 扫描一个完整的源录像，检测人物出现的采样帧。
2. 合并人物事件时间段，按配置保留前后缓冲区。
3. 提取并用 `ffprobe` 检查生成的片段。
4. 处理完成后删除源录像；**没有检测到人物时也会删除源录像**。

检测模型可能漏检；片段检查也不等于逐帧完整性验证。请先备份、只在可丢弃的测试数据上验证，并优先使用 `--dry-run`。不要直接指向唯一一份重要录像。

## 保留的代码

- [主程序](<people_filter.py>)：原始筛选逻辑，未为本次归档修改业务行为。
- [依赖版本](<requirements.txt>)：记录原运行环境中的直接 Python 依赖版本。
- [systemd 示例](<systemd/people-filter.service.example>)：不含原机器用户名、密码或具体摄像头配置，需按实际路径调整；它不是已经启用的服务。

密码文件、虚拟环境、模型权重、录像、日志以及 SQLite 运行数据库均不随仓库上传。

## 原运行环境

- Linux / Python 3.12.13
- NVIDIA GPU、可用于 PyTorch 的 CUDA 环境
- NumPy 2.5.3、PyTorch 2.14.1、Ultralytics 8.4.171
- `ffmpeg` / `ffprobe`，且 FFmpeg 需要支持 `cuda` 硬件解码和 `scale_cuda` 滤镜

Python 依赖之外，仍需自行安装兼容的 NVIDIA 驱动和 FFmpeg。不同 PyTorch 安装源对应的 CUDA 构建可能不同，应按 PyTorch 官方安装说明选择；依赖版本文件不包含驱动。

## 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

默认模型是 `yolo11m.pt`。模型权重需自行获取；Ultralytics 在适用情况下会自动下载。模型和依赖的许可条款由各自项目规定，仓库不重新分发权重。

## 先预览，不删除录像

以下目录仅为示例，输入录像按“输入目录/摄像头名/文件.mkv”布局。输出目录必须在输入目录之外。

```bash
python people_filter.py \
  --input /path/to/test-recordings \
  --output /path/to/test-person-clips \
  --state-db ./preview.sqlite3 \
  --model yolo11m.pt \
  --dry-run --limit 1
```

**预览请使用独立数据库。** 原逻辑会跳过数据库中记录为 `kept_dry_run` 且内容未变的文件；正式处理应使用另一份状态数据库，避免预览记录让文件被跳过。

## 显式运行一次处理

确认测试数据、备份和删除行为后，移除 `--dry-run`：

```bash
python people_filter.py \
  --input /path/to/test-recordings \
  --output /path/to/test-person-clips \
  --state-db ./people-filter.sqlite3 \
  --model yolo11m.pt \
  --limit 1
```

原脚本保留了旧机器的默认绝对路径。迁移后请显式传入 `--input`、`--output`、`--state-db`，不要依赖这些默认值。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--watch` | 关闭 | 持续检查新文件 |
| `--watch-interval` | 600 秒 | 队列处理完后的等待时间 |
| `--sample-fps` | 2 | 普通录像的采样频率 |
| `--keyframe-camera` | 无 | 按摄像头目录名启用关键帧模式，可重复传入 |
| `--keyframe-fps` | 1 | 关键帧模式的采样频率 |
| `--confidence` | 0.15 | 人物检测阈值 |
| `--batch-size` | 64 | 检测批次大小；较大值可能增加显存占用 |
| `--pre-seconds` / `--post-seconds` | 各 30 秒 | 事件前后缓冲 |
| `--min-age-seconds` | 5400 秒 | 只处理足够旧的文件，避免正在录制的数据 |
| `--device` | 0 | CUDA 设备编号 |
| `--dry-run` | 关闭 | 不提取片段、不删除源录像，但仍进行检测并更新预览数据库 |

事件时间由采样帧序号换算；对可变帧率或不规则关键帧间隔的录像应单独核对裁剪时间。片段使用流复制，剪辑边界也可能受关键帧影响。

## 归档与验证范围

本次只做源码归档、语法与提交内容检查，没有重新执行会处理或删除录像的业务流程。原部署的退役由本机服务管理完成；示例服务不会自动启用。若未来重新使用，请先重新验证依赖兼容性、显存占用和数据删除行为。
