#!/bin/bash

# VideoTemp-o3 Rollout 错误快速修复脚本
# 用途：自动应用推荐的修复方案

set -e

echo "=========================================="
echo "  VideoTemp-o3 Rollout 错误修复工具"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. 检查当前配置
echo -e "${BLUE}[1/5] 检查当前配置...${NC}"
if [ ! -f "rl/rollout.sh" ]; then
    echo -e "${RED}✗ 找不到 rl/rollout.sh${NC}"
    exit 1
fi

CURRENT_MEM=$(grep "vllm_gpu_memory_utilization" rl/rollout.sh | grep -oP '\d+\.\d+')
echo -e "${GREEN}✓ 当前 GPU 内存利用率: ${CURRENT_MEM}${NC}"

# 2. 备份原配置
echo ""
echo -e "${BLUE}[2/5] 备份原配置...${NC}"
if [ ! -f "rl/rollout.sh.backup" ]; then
    cp rl/rollout.sh rl/rollout.sh.backup
    echo -e "${GREEN}✓ 已备份到 rl/rollout.sh.backup${NC}"
else
    echo -e "${YELLOW}⚠ 备份文件已存在，跳过${NC}"
fi

# 3. 应用修复方案 1：降低 GPU 内存利用率
echo ""
echo -e "${BLUE}[3/5] 应用修复方案 1：降低 GPU 内存利用率...${NC}"
sed -i 's/--vllm_gpu_memory_utilization 0.75/--vllm_gpu_memory_utilization 0.6/g' rl/rollout.sh
NEW_MEM=$(grep "vllm_gpu_memory_utilization" rl/rollout.sh | grep -oP '\d+\.\d+')
echo -e "${GREEN}✓ GPU 内存利用率已更新: ${CURRENT_MEM} → ${NEW_MEM}${NC}"

# 4. 显示修改内容
echo ""
echo -e "${BLUE}[4/5] 修改内容预览...${NC}"
echo -e "${YELLOW}修改前:${NC}"
grep "vllm_gpu_memory_utilization" rl/rollout.sh.backup
echo -e "${YELLOW}修改后:${NC}"
grep "vllm_gpu_memory_utilization" rl/rollout.sh

# 5. 提供后续建议
echo ""
echo -e "${BLUE}[5/5] 修复完成！${NC}"
echo ""
echo -e "${GREEN}✓ 修复已应用${NC}"
echo ""
echo "=========================================="
echo "  后续步骤"
echo "=========================================="
echo ""
echo "1. 运行 rollout："
echo -e "   ${YELLOW}bash rl/rollout.sh${NC}"
echo ""
echo "2. 监控 GPU 状态（在另一个终端）："
echo -e "   ${YELLOW}watch -n 1 nvidia-smi${NC}"
echo ""
echo "3. 监控日志（在另一个终端）："
echo -e "   ${YELLOW}tail -f logs/baseline_rollout.log${NC}"
echo ""
echo "4. 如果仍然失败，尝试升级 vLLM："
echo -e "   ${YELLOW}pip install vllm==0.10.2 --force-reinstall${NC}"
echo ""
echo "=========================================="
echo ""
echo "详细分析请查看: ROLLOUT_ERROR_ANALYSIS.md"
echo ""
