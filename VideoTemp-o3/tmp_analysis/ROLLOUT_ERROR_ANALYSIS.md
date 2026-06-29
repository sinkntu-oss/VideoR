# Rollout 错误分析与解决方案

## 📋 错误概述

在 `logs/baseline_rollout.log` 中发现的主要错误：

### 1. **vLLM Engine 崩溃** ❌
```
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue
RuntimeError: cancelled
```

**发生时间**: 14:27:36 - 14:27:42

**根本原因**: vLLM 的 EngineCore 进程在处理请求时崩溃，导致共享内存通信中断

### 2. **Swift 推理失败** ❌
```
TypeError: 'NoneType' object is not iterable
File "/root/miniconda3/envs/tempo3/lib/python3.12/site-packages/swift/llm/infer/rollout.py", line 531, in infer
    all_outputs = list(chain.from_iterable(all_outputs))
```

**原因**: vLLM 引擎崩溃导致返回 `None`，Swift 无法处理

### 3. **多进程通信超时** ❌
```
RuntimeError: cancelled
File "/root/miniconda3/envs/tempo3/lib/python3.12/site-packages/vllm/distributed/device_communicators/shm_broadcast.py", line 455
```

**原因**: 共享内存队列通信被取消，可能是由于：
- GPU 内存不足
- 张量并行通信失败
- 工作进程崩溃

---

## 🔍 问题诊断

### 当前配置
```bash
--vllm_tensor_parallel_size 2      # 使用 2 个 GPU
--vllm_gpu_memory_utilization 0.75 # GPU 内存利用率 75%
--vllm_max_model_len 40960         # 最大序列长度
--vllm_limit_mm_per_prompt {"image": 1, "video": 10}  # 多模态限制
```

### 可能的原因

1. **GPU 内存压力过大**
   - 模型大小：Qwen2.5-VL (32B)
   - 张量并行：2 个 GPU
   - 内存利用率：75%
   - 视频处理：每个样本最多 10 个视频

2. **vLLM 版本不兼容**
   ```
   UserWarning: TRL currently only supports vLLM version `0.10.2`. 
   You have version 0.11.0+cu129 installed.
   ```

3. **多轮对话复杂性**
   - `--max_turns 3`：最多 3 轮对话
   - 每轮都需要处理视频和文本
   - 累积的上下文长度可能超过限制

4. **共享内存配置不足**
   - 张量并行通信依赖共享内存
   - 可能需要增加 `/dev/shm` 大小

---

## ✅ 解决方案

### 方案 1: 降低 GPU 内存利用率（推荐）

**修改 `rl/rollout.sh`**:

```bash
# 原配置
--vllm_gpu_memory_utilization 0.75

# 新配置（降低到 60%）
--vllm_gpu_memory_utilization 0.6
```

**优点**:
- 立即生效，无需重新编译
- 减少 OOM 风险
- 保持张量并行

**缺点**:
- 吞吐量可能下降 15-20%

---

### 方案 2: 减少张量并行度

**修改 `rl/rollout.sh`**:

```bash
# 原配置
--vllm_tensor_parallel_size 2

# 新配置（单 GPU）
--vllm_tensor_parallel_size 1
```

**优点**:
- 消除多进程通信开销
- 更稳定的内存管理

**缺点**:
- 单个 GPU 需要 32GB+ 显存
- 吞吐量下降 50%

---

### 方案 3: 升级 vLLM 版本（最佳）

```bash
# 卸载当前版本
pip uninstall vllm -y

# 安装兼容版本
pip install vllm==0.10.2
```

**优点**:
- 解决版本不兼容问题
- 可能修复已知的 bug
- 官方支持

**缺点**:
- 需要重新启动服务
- 可能需要调整其他参数

---

### 方案 4: 减少视频处理复杂度

**修改 `rl/rollout.sh`**:

```bash
# 原配置
--vllm_limit_mm_per_prompt {"image": 1, "video": 10}

# 新配置（减少视频数量）
--vllm_limit_mm_per_prompt {"image": 1, "video": 5}
```

**优点**:
- 减少内存占用
- 加快处理速度

**缺点**:
- 可能影响视频理解质量

---

### 方案 5: 增加共享内存（系统级）

```bash
# 查看当前共享内存大小
df -h /dev/shm

# 临时增加（重启后失效）
mount -o remount,size=16G /dev/shm

# 永久修改（编辑 /etc/fstab）
tmpfs /dev/shm tmpfs size=16G,defaults 0 0
```

---

## 🎯 推荐执行步骤

### 第一步：快速修复（立即尝试）

修改 `rl/rollout.sh`，降低 GPU 内存利用率：

```bash
# 找到这一行
--vllm_gpu_memory_utilization 0.75

# 改为
--vllm_gpu_memory_utilization 0.6
```

然后重新运行：
```bash
bash rl/rollout.sh
```

### 第二步：如果仍然失败

升级 vLLM 版本：

```bash
pip install vllm==0.10.2 --force-reinstall
```

### 第三步：如果还是有问题

同时应用多个优化：

```bash
# 修改 rl/rollout.sh
--vllm_gpu_memory_utilization 0.5
--vllm_tensor_parallel_size 1
--vllm_limit_mm_per_prompt {"image": 1, "video": 5}
```

---

## 📊 性能对比

| 配置 | GPU 内存 | 吞吐量 | 稳定性 | 推荐度 |
|------|---------|--------|--------|--------|
| 原配置 (0.75, TP=2) | 高 | 高 | ❌ 低 | ❌ |
| 方案 1 (0.6, TP=2) | 中 | 中 | ✅ 高 | ⭐⭐⭐ |
| 方案 2 (0.75, TP=1) | 高 | 低 | ✅ 高 | ⭐⭐ |
| 方案 3 (vLLM 0.10.2) | 高 | 高 | ✅ 高 | ⭐⭐⭐⭐ |
| 方案 4 (video=5) | 中 | 中 | ✅ 高 | ⭐⭐⭐ |

---

## 🔧 完整修复脚本

创建 `fix_rollout.sh`：

```bash
#!/bin/bash

echo "=== VideoTemp-o3 Rollout 错误修复 ==="

# 1. 备份原配置
cp rl/rollout.sh rl/rollout.sh.backup
echo "✓ 已备份原配置到 rl/rollout.sh.backup"

# 2. 修改 GPU 内存利用率
sed -i 's/--vllm_gpu_memory_utilization 0.75/--vllm_gpu_memory_utilization 0.6/g' rl/rollout.sh
echo "✓ 已降低 GPU 内存利用率到 60%"

# 3. 升级 vLLM（可选）
# pip install vllm==0.10.2 --force-reinstall

# 4. 显示修改
echo ""
echo "=== 修改内容 ==="
grep "vllm_gpu_memory_utilization" rl/rollout.sh
echo ""
echo "✓ 修复完成！可以运行: bash rl/rollout.sh"
```

运行修复脚本：
```bash
bash fix_rollout.sh
bash rl/rollout.sh
```

---

## 📝 监控建议

运行时监控 GPU 状态：

```bash
# 终端 1：运行 rollout
bash rl/rollout.sh

# 终端 2：监控 GPU
watch -n 1 nvidia-smi

# 终端 3：监控日志
tail -f logs/baseline_rollout.log
```

关键指标：
- GPU 内存使用率 < 90%
- 两个 GPU 的内存使用均衡
- 无 "EngineDeadError" 错误

---

## 🚨 如果问题仍未解决

1. **检查 GPU 状态**
   ```bash
   nvidia-smi
   nvidia-smi -q -d MEMORY
   ```

2. **检查共享内存**
   ```bash
   df -h /dev/shm
   ls -lh /dev/shm/
   ```

3. **查看完整日志**
   ```bash
   tail -500 logs/baseline_rollout.log | grep -i error
   ```

4. **联系技术支持**
   - 提供完整的 `logs/baseline_rollout.log`
   - 提供 `nvidia-smi` 输出
   - 提供当前的 `rl/rollout.sh` 配置

---

## 📚 相关文档

- [vLLM 官方文档](https://docs.vllm.ai/)
- [Swift Rollout 配置](https://github.com/modelscope/swift)
- [Qwen2.5-VL 模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)

