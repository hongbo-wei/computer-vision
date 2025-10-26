# facial-recognition（中文）

[![EN](https://img.shields.io/badge/lang-EN-blue)](README.md) [![中文](https://img.shields.io/badge/语言-中文-red)](README.zh.md)

本文件夹包含使用 `face_recognition`（基于 `dlib`）与 OpenCV 的简单示例脚本，用于演示人脸检测与识别的基本流程。

文件说明

- `face_finder.py` — 在群体照片中找到目标人物（通过人脸嵌入比较）并绘制矩形框。
- `find_faces_in_picture.py` —（辅助）检测并列出单张图片中的人脸位置。
- `images/` — 存放示例图片（例如 `target.jpg`、`group.jpg`）。
- `requirements.txt` — 已固定的 Python 依赖，可以通过 `pip install -r requirements.txt` 安装。

简要说明

- 检测：默认 `face_recognition.face_locations(...)` 使用 HOG（基于方向梯度直方图）的检测方法（在 CPU 上速度快）。
- 嵌入（embedding）：`face_recognition.face_encodings(...)` 使用 dlib 的人脸识别模型（基于 ResNet 的 CNN）来生成每张人脸的向量表示，用于后续比对。

因此该流程混合了传统检测（HOG）和深度学习生成的嵌入（CNN）。如需更高的检测准确率，可以在 `face_locations` 中传入 `model="cnn"` 来使用基于 CNN 的检测器（更慢）。

安装（macOS / zsh）

1. 创建并激活虚拟环境（推荐）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. 使用 `requirements.txt` 安装依赖：

```bash
pip install -r requirements.txt
```

如果在 macOS 上安装 `dlib` 失败，请先安装构建依赖：

```bash
# macOS (Homebrew)
brew install cmake pkg-config
pip install dlib
pip install face_recognition
```

使用方法：

1. 将图片放入 `images/` 文件夹（默认脚本使用 `target.jpg` 和 `group.jpg`）。
2. 运行：

```bash
python face_finder.py
```

脚本做了什么

- 加载“目标”人脸图片并计算其人脸嵌入（由 dlib 的 CNN 模型产生）。
- 加载群体图片，检测人脸位置，为每个位置计算嵌入，并将其与目标嵌入进行比较（`compare_faces`，默认容差 0.6）。
- 若匹配则在该人脸处绘制矩形并标注“Target”，并用 OpenCV 显示或保存结果。

常见调整

- 使用 CNN 检测器（更准确）：

```py
face_locations = face_recognition.face_locations(image, model="cnn")
```

- 无 GUI 环境：将 `cv2.imshow(...)` + `waitKey` 替换为 `cv2.imwrite('out.jpg', image_bgr)` 保存结果。
- 增强鲁棒性：在获取 `known_encoding = face_recognition.face_encodings(...)` 前，先确认返回列表非空，避免索引错误。
- 若需标注所有匹配项，移除匹配后脚本中的 `break`。
- 可通过调整 `compare_faces(..., tolerance=...)` 的 `tolerance` 值控制严格度。

隐私与合规

- 本示例用于本地试验，请确保对被处理的图像拥有合法权限。面部识别具有敏感性，请遵守当地法律法规与隐私最佳实践。

鸣谢

- 本项目使用了 Adam Geitgey 的 `face_recognition`（https://github.com/ageitgey/face_recognition）以及 `dlib` 的模型。

许可证

- 示例代码按原样提供，请根据你的项目选择合适的许可与使用条款。

如果你需要，我可以：

- 将 `face_finder.py` 改为更健壮的版本（添加命令行参数以选择检测器或输出路径，并添加缺失嵌入检查）。
