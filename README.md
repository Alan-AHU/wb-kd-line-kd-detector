# kd-line-kd-detector

识别 Western blot 图中灰黑色条带，输出每条带的纵坐标 `y` 与对应分子量 `kD`。

## 功能

- 自动定位 blot 区域（左下方灰底区域）
- 自动检测泳道中心
- 自动检测各泳道灰黑条带
- 基于左侧刻度线拟合 `log10(kD) = a*y + b`
- 输出 `CSV`、`JSON`、叠加可视化图片

## 安装

```powershell
cd D:\KD\kd-line-kd-detector
python -m pip install -r requirements.txt
```

## 运行

```powershell
cd D:\KD\kd-line-kd-detector
python run.py --image data/input_image.png --output-dir outputs
```

如果不想包含最左侧 marker 泳道（`M`）：

```powershell
python run.py --image data/input_image.png --output-dir outputs --drop-marker-lane
```

## 输出文件

- `outputs/input_image_overlay.png`：检测可视化
- `outputs/input_image_bands.csv`：每条带的结构化结果
- `outputs/input_image_bands.json`：完整元数据与结果

`CSV` 字段：

- `lane_label`：泳道编号（`M` 或 `1..N`）
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
