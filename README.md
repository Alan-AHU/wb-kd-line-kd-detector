# kd-line-kd-detector

识别 Western blot 图中灰黑色条带，输出每条带的纵坐标 `y` 与对应分子量 `kD`。

## 功能

- 自动定位 blot 区域（左下方灰底区域）
- 自动检测泳道中心
- 自动检测各泳道灰黑条带
- 基于左侧刻度线拟合 `log10(kD) = a*y + b`
- 输出 `CSV`、`JSON`、`XLSX`、叠加可视化图片
- 可视化图中自动标点并写出每个点的 `kD` 值
- 横坐标泳道按 `1..N` 编号显示（例如 `1..8`）
- 支持单图和整文件夹批量处理

## 安装

```powershell
cd D:\KD\kd-line-kd-detector
python -m pip install -r requirements.txt
```

## 运行

单图：

```powershell
cd D:\KD\kd-line-kd-detector
python run.py --image data/input_image.png --output-dir outputs
```

批量处理整个文件夹：

```powershell
cd D:\KD\kd-line-kd-detector
python run.py --input-dir "D:\protein-20260109\protein-20260109\image" --output-dir outputs_batch
```

如果不想包含最左侧泳道（按检测顺序的第 1 列）：

```powershell
python run.py --input-dir "D:\protein-20260109\protein-20260109\image" --output-dir outputs_batch --drop-marker-lane
```

## 输出文件

- `outputs/input_image_annotated.png`：检测可视化（点位 + kD文字 + 横坐标 `1..N`）
- `outputs/input_image_bands.csv`：每条带的结构化结果
- `outputs/input_image_bands.json`：完整元数据与结果
- `outputs/input_image_bands.xlsx`：单图 Excel 结果
- `outputs_batch/batch_summary.xlsx`：批量汇总 Excel

`CSV` 字段：

- `lane_x_index`：横坐标泳道编号（从 1 开始）
- `lane_label`：与 `lane_x_index` 一致
- `x`：泳道内横向位置（ROI 坐标）
- `y`：原图纵坐标（像素）
- `kD`：换算得到的分子量
- `prominence`：峰显著性
- `peak_height`：峰强度
- `extrapolated`：是否超出刻度拟合区间

## 测试

```powershell
cd D:\KD\kd-line-kd-detector
python -m pytest -q
```

## 上传 GitHub（无 `git/gh` 方案）

先设置 `GITHUB_TOKEN`（需要 `repo` 权限），再执行：

```powershell
cd D:\KD\kd-line-kd-detector
$env:GITHUB_TOKEN="你的token"
python publish_to_github.py --owner 你的GitHub用户名 --repo kd-line-kd-detector
```

若提示 `Resource not accessible by personal access token`，先在 GitHub 网页手动新建空仓库，再执行：

```powershell
python publish_to_github.py --owner 你的GitHub用户名 --repo 你的仓库名 --skip-create
```
