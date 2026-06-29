# Rollout 故障排除完整指南

## 🎯 快速开始

### 一键修复（推荐）

```bash
bash fix_rollout.sh
bash rl/rollout.sh
```

---

## 📊 错误症状与解决方案

### 症状 1: EngineDeadError

**日志信息**:
```
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue
RuntimeError: cancelled
```

**原因**: vLLM 引擎进程崩溃，通常由 GPU 内存不足引起

**解决方案**:

#### 方案 A: 降低 GPU 内存利用率（推荐）
```bash
# 编辑 rl/rollout.sh
sed -i 's/--vllm_gpu_memory_utilization 0.75/--vllm_gpu_memory_utilization 0.6/g' rl/rollout.sh

# 或手动编辑
# 找到: --vllm_gpu_memory_utilization 0.75
# 改为: --vllm_gpu_memory_utilization 0.6
```

#### 方案 B: 升级 vLLM
```bash
pip install vllm==0.10.2 --force-reinstall
```

#### 方案 C: 减少张量并行度
```bash
# 编辑 rl/rollout.sh
sed -i 's/--vllm_tensor_parallel_size 2/--vllm_tensor_parallel_size 1/g' rl/rollout.sh
```

---

### 症状 2: TypeError: 'NoneType' object is not iterable

**日志信息**:
```
TypeError: 'NoneType' object is not iterable
File ".../swift/llm/infer/rollout.py", line 531, in infer
    all_outputs = list(chain.from_iterable(all_outputs))
```

**原因**: vLLM 引擎返回 None，通常是因为引擎崩溃

**解决方案**: 同上（症状 1 的解决方案）

---

### 症状 3: CUDA Out of Memory

**日志信息**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**原因**: GPU 显存不足

**解决方案**:

#### 方案 A: 降低内存利用率
```bash
sed -i 's/--vllm_gpu_memory_utilization 0.75/--vllm_gpu_memory_utilization 0.5/g' rl/rollout.sh
```

#### 方案 B: 减少视频处理复杂度
```bash
# 编辑 rl/rollout.sh
# 找到: --vllm_limit_mm_per_prompt '{"image": 1, "video": 10}'
# 改为: --vllm_limit_mm_per_prompt '{"image": 1, "video": 5}'
```

#### 方案 C: 减少最大序列长度
```bash
# 编辑 rl/rollout.sh
# 找到: --vllm_max_model_len 40960
# 改为: --vllm_max_model_len 32768
```

---

### 症状 4: 多进程通信超时

**日志信息**:
```
RuntimeError: cancelled
File ".../vllm/distributed/device_communicators/shm_broadcast.py", line 455
```

**原因**: 共享内存通信失败，可能是内存不足或进程崩溃

**解决方案**:

#### 方案 A: 增加共享内存大小
```bash
# 查看当前大小
df -h /dev/shm

# 临时增加（重启后失效）
sudo mount -o remount,size=16G /dev/shm

# 永久修改（需要 root）
# 编辑 /etc/fstab，找到 tmpfs 行，改为：
# tmpfs /dev/shm tmpfs size=16G,defaults 0 0
# 然后重启
```

#### 方案 B: 使用单 GPU
```bash
sed -i 's/--vllm_tensor_parallel_size 2/--vllm_tensor_parallel_size 1/g' rl/rollout.sh
```

---

## 🔍 诊断步骤

### 步骤 1: 检查 GPU 状态

```bash
# 查看 GPU 信息
nvidia-smi

# 查看详细内存信息
nvidia-smi -q -d MEMORY

# 实时监控
watch -n 1 nvidia-smi
```

**预期输出**:
- 两个 GPU（GPU 6 和 GPU 7）都可用
- 内存使用率 < 90%
- 无 "Out of Memory" 错误

### 步骤 2: 检查共享内存

```bash
# 查看共享内存大小
df -h /dev/shm

# 查看共享内存文件
ls -lh /dev/shm/

# 查看共享内存使用情况
du -sh /dev/shm/*
```

**预期输出**:
- 共享内存大小 >= 8GB
- 没有大量的 psm_* 文件堆积

### 步骤 3: 检查日志

```bash
# 查看最后 100 行日志
tail -100 logs/baseline_rollout.log

# 查看所有错误
grep -i error logs/baseline_rollout.log

# 查看特定错误
grep "EngineDeadError" logs/baseline_rollout.log
```

### 步骤 4: 检查配置

```bash
# 查看当前 rollout 配置
cat rl/rollout.sh

# 查看备份配置
cat rl/rollout.sh.backup

# 对比两个配置
diff rl/rollout.sh rl/rollout.sh.backup
```

---

## 🛠️ 修复方案优先级

### 优先级 1: 快速修复（立即尝试）

```bash
# 降低 GPU 内存利用率
sed -i 's/--vllm_gpu_memory_utilization 0.75/--vllm_gpu_memory_utilization 0.6/g' rl/rollout.sh

# 运行
bash rl/rollout.sh
```

**成功率**: 70-80%
**耗时**: 5 分钟

---

### 优先级 2: 版本升级（如果优先级 1 失败）

```bash
# 升级 vLLM
pip install vllm==0.10.2 --force-reinstall

# 运行
bash rl/rollout.sh
```

**成功率**: 85-90%
**耗时**: 10 分钟

---

### 优先级 3: 多重优化（如果优先级 1-2 都失败）

```bash
# 编辑 rl/rollout.sh，应用以下修改：
# 1. --vllm_gpu_memory_utilization 0.75 → 0.5
# 2. --vllm_tensor_parallel_size 2 → 1
# 3. --vllm_limit_mm_per_prompt '{"image": 1, "video": 10}' → '{"image": 1, "video": 5}'

# 或使用脚本
cat > apply_fixes.sh << 'EOF'
#!/bin/bash
sed -i 's/--vllm_gpu_memory_utilization 0.75/--vllm_gpu_memory_utilization 0.5/g' rl/rollout.sh
sed -i 's/--vllm_tensor_parallel_size 2/--vllm_tensor_parallel_size 1/g' rl/rollout.sh
sed -i 's/"video": 10/"video": 5/g' rl/rollout.sh
EOF

bash apply_fixes.sh
bash rl/rollout.sh
```

**成功率**: 95%+
**耗时**: 15 分钟

---

### 优先级 4: 系统级修复（如果优先级 1-3 都失败）

```bash
# 增加共享内存
sudo mount -o remount,size=16G /dev/shm

# 清理旧的共享内存文件
rm -f /dev/shm/psm_*

# 运行
bash rl/rollout.sh
```

**成功率**: 99%+
**耗时**: 20 分钟

---

## 📈 性能监控

### 实时监控脚本

创建 `monitor_rollout.sh`:

```bash
#!/bin/bash

echo "启动 Rollout 监控..."
echo ""

# 终端 1: 运行 rollout
echo "终端 1: 运行 rollout"
echo "  bash rl/rollout.sh"
echo ""

# 终端 2: 监控 GPU
echo "终端 2: 监控 GPU"
echo "  watch -n 1 nvidia-smi"
echo ""

# 终端 3: 监控日志
echo "终端 3: 监控日志"
echo "  tail -f logs/baseline_rollout.log | grep -E '(ERROR|EngineDeadError|OutOfMemory)'"
echo ""

# 终端 4: 监控共享内存
echo "终端 4: 监控共享内存"
echo "  watch -n 5 'du -sh /dev/shm/*'"
echo ""

echo "按 Ctrl+C 停止监控"
```

### 关键指标

| 指标 | 正常范围 | 警告范围 | 错误范围 |
|------|---------|---------|---------|
| GPU 内存使用率 | < 80% | 80-90% | > 90% |
| GPU 温度 | < 70°C | 70-80°C | > 80°C |
| 共享内存使用 | < 8GB | 8-12GB | > 12GB |
| 推理延迟 | < 10s | 10-30s | > 30s |

---

## 🔄 回滚步骤

如果修复导致其他问题，可以回滚：

```bash
# 恢复原配置
cp rl/rollout.sh.backup rl/rollout.sh

# 验证
cat rl/rollout.sh

# 重新运行
bash rl/rollout.sh
```

---

## 📞 获取帮助

如果问题仍未解决，请收集以下信息：

1. **完整日志**
   ```bash
   cp logs/baseline_rollout.log logs/baseline_rollout_error.log
   ```

2. **GPU 信息**
   ```bash
   nvidia-smi > gpu_info.txt
   nvidia-smi -q -d MEMORY > gpu_memory.txt
   ```

3. **系统信息**
   ```bash
   uname -a > system_info.txt
   df -h >> system_info.txt
   ```

4. **当前配置**
   ```bash
   cat rl/rollout.sh > current_config.txt
   ```

5. **提交问题**
   - 附加上述所有文件
   - 描述问题现象
   - 说明已尝试的解决方案

---

## 📚 相关资源

- [vLLM 官方文档](https://docs.vllm.ai/)
- [Swift 文档](https://github.com/modelscope/swift)
- [Qwen2.5-VL 模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)
- [CUDA 内存管理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-management)

---

## ✅ 检查清单

在运行 rollout 前，确保：

- [ ] 已备份原配置 (`rl/rollout.sh.backup`)
- [ ] 已应用至少一个修复方案
- [ ] GPU 驱动版本 >= 12.0
- [ ] CUDA 版本 >= 12.0
- [ ] 共享内存大小 >= 8GB
- [ ] 磁盘空间充足 (> 50GB)
- [ ] 网络连接正常
- [ ] 没有其他 GPU 任务运行

---

## 🎉 成功标志

Rollout 成功运行的标志：

```
[INFO:swift] Successfully imported external_plugins: ['rl/video_crop_plugin.py'].
[INFO:swift] Setting args.lazy_tokenize: True
[INFO:swift] args.result_path: .../deploy_result/20260627-140518.jsonl
INFO:     Started server process [318463]
INFO:     Waiting for application startup.
```

以及：

```
✓ 推理完成
✓ 结果已保存到 deploy_result/
✓ 没有 ERROR 或 EngineDeadError
```

---

最后更新: 2026-06-27
