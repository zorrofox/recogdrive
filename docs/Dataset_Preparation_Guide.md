# ReCogDrive 数据准备指南

本文档详细说明了 ReCogDrive 训练所需的数据集准备流程，包括数据集的分类、获取方式以及配置方法。

## 数据集概览

ReCogDrive 的训练（特别是 Stage 1 预训练）依赖于两类数据：

1.  **ReCogDrive 专属数据集**：基于 NAVSIM 模拟器生成，包含轨迹和自动标注的问答数据。
2.  **外部 QA 数据集**：13 个公开的自动驾驶问答数据集，用于增强模型的通用理解能力。

---

## 1. ReCogDrive 专属数据集 (可自主生成)

这部分数据不需要下载，而是利用项目提供的脚本基于 NAVSIM 环境生成。

### 前置条件
*   已安装并配置好 NAVSIM 和 NuPlan 环境。
*   已下载 NuPlan 的地图和传感器基础数据（使用 `download/` 目录下的脚本）。

### 生成步骤
在项目根目录下运行以下命令：

```bash
cd ./scripts

# 1. 生成轨迹数据集 (Trajectory Dataset)
sh generate_dataset/generate_internvl_dataset.sh

# 2. 生成自动标注数据集 (Auto-labeled Pipeline Dataset)
# 注意：运行此脚本前需部署 VLM (如 vllm 或 Sglang) 用于自动生成标签
sh generate_dataset/generate_internvl_dataset_pipeline.sh
```

---

## 2. 外部 QA 数据集 (需自行下载)

由于隐私和许可原因，项目方不提供直接下载链接。您需要从以下官方渠道获取这 13 个数据集。

| 数据集名称 | 简介 | 官方获取渠道 |
| :--- | :--- | :--- |
| **DriveLM** | 基于 nuScenes 的图结构问答（感知/预测/规划） | [GitHub - OpenDriveLab/DriveLM](https://github.com/OpenDriveLab/DriveLM) |
| **LingoQA** | 自动驾驶视频问答基准 | [GitHub - wayveai/LingoQA](https://github.com/wayveai/LingoQA) |
| **CODA-LM** | 关注长尾场景（Corner Cases）的视觉语言数据 | [GitHub - AIR-DISCOVER/CODA-LM](https://github.com/AIR-DISCOVER/CODA-LM) |
| **NuScenes-QA** | 基于 nuScenes 的大规模视觉问答 | [GitHub - qiantianwen/NuScenes-QA](https://github.com/qiantianwen/NuScenes-QA) |
| **Talk2Car** | 自然语言命令与轨迹/目标对应数据 | [GitHub - talk2car/Talk2Car](https://github.com/talk2car/Talk2Car) |
| **DriveGPT4** | LLM 生成的驾驶场景描述与问答 | [Project Page](https://tonyxuqaq.github.io/projects/DriveGPT4/) |
| **SUTD-TrafficQA** | 交通场景视频问答 | [GitHub - SUTD-TrafficQA](https://github.com/SUTD-TrafficQA/SUTD-TrafficQA) |
| **MAPLM** | 专注于地图信息的驾驶问答 | [GitHub - LLVM-AD/MAPLM](https://github.com/LLVM-AD/MAPLM) |
| **NuInstruct** | 基于 nuScenes 的指令微调数据 | [GitHub - msight-tech/NuInstruct](https://github.com/msight-tech/NuInstruct) |
| **OmniDrive** | 多视角图像与 3D 信息问答 | [GitHub - NVlabs/OmniDrive](https://github.com/NVlabs/OmniDrive) |
| **DRAMA** | 驾驶风险定位与解释 | [Honda Research Institute](https://usa.honda-ri.com/drama) |
| **Senna** | 面向规划的自动标注数据 | 需查阅 Senna 论文或联系作者获取 |
| **LLaVA** | 通用视觉指令微调数据 (LLaVA-Instruct-150K) | [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) |

---

## 3. 数据集配置

下载并准备好数据后，您需要修改配置文件以指向您的本地路径。

**配置文件位置**: `internvl_chat/shell/data_info/recogdrive_pretrain.json`

**配置示例**:
打开该 JSON 文件，找到对应的数据集条目，修改 `"root"` 和 `"annotation"` 字段。

```json
{
    "DriveLM": {
      "root": "/您的本地路径/nuscenes",  <-- 修改这里
      "annotation": "/您的本地路径/DriveLM/output_nus.jsonl", <-- 修改这里
      "data_augment": false,
      "repeat_time": 1,
      "length": 4072
    },
    ...
}
```

## 4. 训练阶段的数据需求

*   **Stage 1 (VLM 预训练)**:
    *   **完整复现**: 需要 **ReCogDrive 数据集** + **13 个外部 QA 数据集**。
    *   **最小可行性**: 仅使用 **ReCogDrive 数据集**（需在 JSON 配置中删除未下载的外部数据集）。这能跑通流程，但模型泛化能力会下降。

*   **Stage 2 (规划器训练)**:
    *   **仅需**: **ReCogDrive 数据集**。
    *   **建议**: 如果不想跑 Stage 1，可以直接下载作者提供的预训练 VLM 权重，配合自己生成的 ReCogDrive 数据集直接开始 Stage 2。
