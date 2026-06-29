# VideoTemp-o3 项目说明

> **论文**: [VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking-with-Videos](https://arxiv.org/abs/2602.07801) (ICML 2026)
>
> **来源**: 快手 Kwai-Keye | [GitHub](https://github.com/Kwai-Keye/VideoTemp-o3) | [模型权重](https://huggingface.co/Kwai-Keye/VideoTemp-o3) | [数据集](https://huggingface.co/datasets/Kwai-Keye/VideoTemp-o3) | [Benchmark](https://huggingface.co/datasets/Kwai-Keye/VideoTemp-Bench)

---

## 目录结构

```
VideoTemp-o3/
├── README.md                        # 原始英文 README
├── README_zh.md                     # 本文件（中文说明）
├── README_SCENE.md                  # 场景说明文档
├── requirement.txt                  # Python 依赖
├── setup_env.sh                     # 一键环境安装脚本（conda 环境 Tempo3）
├── setup_data.sh                    # 一键数据组织脚本（从 datasets/VideoTemp-o3 建立软链接）
├── run_eval_videomme.sh             # Video-MME 评测快捷脚本
│
├── figs/                            # 论文图片
│   └── main.png
│
├── sft/                             # SFT 训练
│   ├── sft.sh                       # SFT 训练启动脚本
│   ├── sft_events.sh                # SFT 事件驱动训练脚本
│   ├── loss_scale_plugin.py         # 自定义 loss scale 插件
│   ├── data/                        # SFT 训练数据（标注 jsonl，均为软链接）
│   │   ├── wo_tool_call/            # 冷启动数据（不含工具调用）
│   │   │   ├── activitynet.jsonl    -> datasets/VideoTemp-o3/sft/activitynet.jsonl
│   │   │   ├── charades.jsonl       -> datasets/VideoTemp-o3/sft/charades.jsonl
│   │   │   ├── vidchapters.jsonl    -> datasets/VideoTemp-o3/sft/vidchapters.jsonl
│   │   │   ├── video_r1_image_mc.jsonl -> datasets/VideoTemp-o3/sft/video_r1_image_mc.jsonl
│   │   │   └── video_r1_video.jsonl -> datasets/VideoTemp-o3/sft/video_r1_video.jsonl
│   │   └── wi_tool_call/            # 含工具调用数据
│   │       ├── activitynet.jsonl    -> datasets/VideoTemp-o3/sft_tool_call/activitynet.jsonl
│   │       ├── longvila.jsonl       -> datasets/VideoTemp-o3/sft_tool_call/longvila.jsonl
│   │       └── qvhighlight.jsonl    -> datasets/VideoTemp-o3/sft_tool_call/qvhighlight.jsonl
│   ├── data_events/                 # SFT 事件驱动训练数据
│   │   ├── wi_tool_call/            # 含工具调用数据
│   │   └── wo_tool_call/            # 不含工具调用数据
│   └── ckpt/                        # SFT 训练 checkpoint 输出目录
│       └── test/                    # 测试运行产出（v0/v1/v2-*）
│
├── rl/                              # RL (GRPO) 训练
│   ├── grpo.sh                      # GRPO 训练启动脚本（GPU 0-5）
│   ├── grpo_events.sh               # GRPO 事件驱动训练脚本
│   ├── rollout.sh                   # Rollout 推理引擎启动脚本（GPU 6-7）
│   ├── rollout_events.sh            # Rollout 事件驱动推理脚本
│   ├── video_crop_plugin.py         # 视频裁剪工具调用插件
│   ├── video_event_plugin.py        # 视频事件处理插件
│   ├── data/                        # RL 训练数据（标注 jsonl，均为软链接）
│   │   ├── qa.jsonl                 -> datasets/VideoTemp-o3/rl/qa-1k.jsonl
│   │   └── grounding.jsonl          -> datasets/VideoTemp-o3/rl/grounding.jsonl
│   ├── data_events/                 # RL 事件驱动训练数据
│   └── temp_videos/                 # 临时视频存储（事件驱动生成）
│       └── [timestamp]/             # 按时间戳组织的视频目录
│
├── eval/                            # 评测代码
│   ├── 7b_deploy_1024.sh            # vLLM 推理引擎部署脚本
│   ├── score.py                     # 统一评分脚本
│   ├── utils.py                     # 评测工具函数
│   ├── videomme/                    # Video-MME 评测
│   │   ├── videomme.py
│   │   ├── data/                    # 评测 parquet 数据
│   │   │   └── videomme/
│   │   └── agent_runs/              # 评测运行结果
│   ├── mlvu/                        # MLVU 评测
│   │   └── mlvu.py
│   ├── lvbench/                     # LVBench 评测
│   │   └── lvbench.py
│   ├── videommmu/                   # Video-MMMU 评测
│   │   └── videommmu.py
│   └── videotemp/                   # VideoTemp-Bench 评测
│       ├── videotemp.py             # MCQ 评测
│       └── videotemp-g.py           # Grounding 评测
│
├── scripts/                         # 辅助脚本
│   ├── analyze_tsm_heatmap.py       # TSM 热力图分析脚本
│   ├── convert_annotations.py       # 标注转换脚本
│   ├── preprocess_scenes.py         # 场景预处理脚本
│   ├── prepare_event_data.sh        # 事件数据准备脚本
│   └── tsm_heatmap.png              # TSM 热力图示例
│
├── tmp_analysis/                    # 临时分析目录
│   ├── fix_rollout.sh               # Rollout 修复脚本
│   ├── ROLLOUT_ERROR_ANALYSIS.md    # Rollout 错误分析文档
│   ├── ROLLOUT_FIX_SUMMARY.md       # Rollout 修复总结
│   └── ROLLOUT_TROUBLESHOOTING.md   # Rollout 故障排除指南
│
│── # ===== 以下为视频数据目录（软链接） =====
│
├── sft_videos/                      -> datasets/VideoTemp-o3/sft_videos（SFT 训练视频总目录）
├── rl_videos/                       -> datasets/VideoTemp-o3/rl_videos （RL 训练视频总目录）
│
│── # ===== 以下为顶层视频快捷链接（指向 sft_videos/ 下对应子目录） =====
│
├── ActivityNet/                     -> sft_videos/ActivityNet
├── Charades_v1/                     -> sft_videos/Charades_v1
├── cropped_video/                   -> sft_videos/cropped_video
├── LongVILA/                        -> sft_videos/LongVILA
├── QVhilights/                      -> sft_videos/QVhilights
├── VidChapters/                     -> sft_videos/VidChapters
└── Video-R1-data/                   -> sft_videos/Video-R1-data
```

> **说明**: 项目根目录下的 `ActivityNet/`、`Charades_v1/` 等是指向 `sft_videos/` 子目录的软链接。
> jsonl 中的视频路径（如 `ActivityNet/videos/v_xxx.mp4`）从项目根目录开始解析，
> 因此这些顶层软链接确保了训练时视频路径能正确找到。

---

## 训练视频数据

训练视频存放在 `datasets/VideoTemp-o3/` 目录下（通过 `setup_data.sh` 软链接到项目内），来源如下：

| 目录 | 视频来源 | 说明 |
|------|---------|------|
| `sft_videos/ActivityNet/` | [ActivityNet](https://cs.stanford.edu/people/ranjaykrishna/densevid/) | SFT 训练视频 |
| `sft_videos/Charades_v1/` | [Charades](https://github.com/jiyanggao/TALL) | SFT 训练视频 |
| `sft_videos/LongVILA/` | [LongVILA](https://huggingface.co/datasets/LongVILA/longvila_sft_dataset) | SFT 训练视频 |
| `sft_videos/QVhilights/` | [QvHighlight](https://github.com/jayleicn/moment_detr) | SFT 训练视频 |
| `sft_videos/VidChapters/` | [VidChapters-7M](https://github.com/antoyang/VidChapters) | SFT 训练视频 |
| `sft_videos/Video-R1-data/` | [Video-R1](https://huggingface.co/datasets/Video-R1/Video-R1-data) | SFT 训练视频 |
| `sft_videos/cropped_video/` | 裁剪后的视频片段 | SFT 训练视频 |
| `rl_videos/` | RL 训练用视频 | 待解压 tar.gz |

---

## 快速开始

```bash
# 1. 安装环境
bash setup_env.sh

# 2. 组织数据（从 datasets/VideoTemp-o3 建立软链接）
bash setup_data.sh

# 3. SFT 训练
conda activate Tempo3
bash sft/sft.sh

# 4. RL 训练（先启动 rollout，再启动 grpo）
bash rl/rollout.sh   # 终端 1
bash rl/grpo.sh      # 终端 2

# 5. 评测（以 Video-MME 为例）
bash eval/7b_deploy_1024.sh          # 部署推理引擎
python eval/videomme/videomme.py     # 运行推理
python eval/score.py videomme        # 计算分数
```

---

## 评测 Benchmark 下载

| Benchmark | 下载地址 | 存放位置 |
|-----------|---------|---------|
| Video-MME | https://huggingface.co/datasets/lmms-lab/Video-MME | `eval/videomme/data/` |
| MLVU | https://huggingface.co/datasets/MLVU/MLVU_Test | `eval/mlvu/data/` |
| Video-MMMU | https://huggingface.co/datasets/lmms-lab/VideoMMMU | `eval/videommmu/data/` |
| LVBench | https://huggingface.co/datasets/zai-org/LVBench | `eval/lvbench/data/` |
| VideoTemp-Bench | https://huggingface.co/datasets/Kwai-Keye/VideoTemp-Bench | `eval/videotemp/data/` |
