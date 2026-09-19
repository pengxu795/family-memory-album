# 模型权重许可来源清单（出厂合规依据）

> 2026-09-09 建立（豆包顾问提醒 + A4 终裁回退 SFace 触发）。
> 原则：**代码许可 ≠ 权重许可**。出厂分发以「权重训练数据许可」为准，MIT/Apache 仓库
> 不自动覆盖其预训练权重。随产品出厂的模型必须整行无红字。

## 出厂随分发（整行绿）

| 模型 | 用途 | 代码许可 | 权重/训练数据许可 | 出处 |
|---|---|---|---|---|
| YuNet (face_detection_yunet_2023mar) | 人脸检测 | OpenCV Zoo Apache-2.0 | Apache-2.0（OpenCV Zoo 官方分发） | github.com/opencv/opencv_zoo |
| SFace (face_recognition_sface_2021dec) | 人脸识别（**出厂默认**，2026-09-09 终裁回退） | OpenCV Zoo Apache-2.0 | Apache-2.0（OpenCV Zoo 官方分发；训练集 MS1M 衍生，OpenCV 以 Apache-2.0 再分发） | 同上 |
| SigLIP2 base-patch16-224 | 图像检索/相似去重 | HF Apache-2.0 | apache-2.0（Google 官方 HF 仓库） | huggingface.co/google/siglip2-base-patch16-224 |
| U²-Netp (u2netp.onnx) | 显著性裁切 | rembg 同源导出，项目 MIT/Apache-2.0 | Apache-2.0（原作者开源） | github.com/xuebinqin/U-2-Net |
| RapidOCR PP-OCRv4 | 照片文字 OCR | Apache-2.0 | Apache-2.0（PaddlePaddle 官方模型） | github.com/RapidAI/RapidOCR |
| LAION 美学线性头 (sa_0_4_vit_b_32_linear) | 美学评分 | **MIT**（官方源仓库 LAION-AI/aesthetic-predictor，2026-09-09 经 GitHub API 核实 license.spdx_id=mit） | MIT（可商用分发；本地 .pth 须溯源到官方仓库或其 HF 官方镜像，勿用不明第三方权重）。备选：改进版 MLP 头 sac+logos+ava1-l14-linearMSE（christophschuhmann/improved-aesthetic-predictor）为 Apache-2.0，追求更稳可直接换用 | github.com/LAION-AI/aesthetic-predictor |

## 仅本地自用（权重不进镜像、不随产品分发）

| 模型 | 用途 | 代码许可 | 权重/训练数据许可 | 红线 |
|---|---|---|---|---|
| AdaFace IR-18 (WebFace4M) | 人脸识别（本地现役，512 维） | 仓库 MIT | **WebFace4M 非商业研究许可**——仓库 MIT 只覆盖代码，不覆盖权重 | 本地自用可；出厂分发禁止 |
| ArcFace-w600k_r50 (insightface) | 人脸识别备选 | 仓库 MIT（代码自实现无 pip 依赖） | insightface 官方 zoo 权重**非商业研究许可** | 同上 |
| genderage.onnx (insightface zoo) | 性别年龄 | 同上 | insightface 官方 zoo **非商业研究许可** | 同上 |
| 2d106det (insightface zoo) | 106 关键点/表情 | 同上 | 同上 | 同上 |

## 裁定依据

- 2026-09-09 晚终裁：AdaFace pair-F1=0.511 < SFace 0.542（旧口径 0.580 作废）+ WebFace4M
  权重非商业 → **出厂默认回退 SFace**，本地库保持 AdaFace（plist FF_FACE_BACKEND=adaface）。
  详见 eval/threshold_addendum_20260909.md。
- LAION 美学头待核实项已于 2026-09-09 闭环（GitHub API 实证 MIT），出厂前检查项清零；
  唯一保留动作：确认本地 sa_0_4_vit_b_32_linear.pth 的下载来源可溯源到官方仓库。
