# 钢轨异常检测 - 快速开始

## 🚀 一键启动训练

### 本地环境（Windows）
```bash
cd G:\SteelRailWay
scripts\train_rail_cam1.bat
```

### 服务器环境（Linux）

#### 单个 Cam 训练
```bash
cd /home/root123/LF/WYM/SteelRailWay
bash scripts/train_rail_cam1_server_preload.sh
```

#### 串行训练多个 Cam（推荐）⭐ 新增
```bash
# 训练所有 8 个 Cam
bash scripts/train_all_cams_server.sh

# 训练指定的 Cam
bash scripts/train_all_cams_server.sh 1 2 3

# 只训练 Cam 1
bash scripts/train_all_cams_server.sh 1
```

---

## 📊 数据集统计

### 本地环境
| 项目 | 数量 |
|---|---|
| 训练集图像 | 294 × 8 = 2,352 张 |
| 训练集 Patch | 2,352 × 7 = 16,464 个 |
| 测试集图像 | 约 21 张（cam1） |
| 测试集 Patch | 21 × 7 = 147 个 |

### 服务器环境（10% 采样，8:1:1 划分）⭐ 更新
| 项目 | 数量 |
|---|---|
| 原始训练集 | 29,400 × 8 = 235,200 张 |
| 采样后（10%） | 2,940 × 8 = 23,520 张 |
| **Train（80%）** | 2,352 × 8 = 18,816 张（正常样本） |
| **Val（10%）** | 294 × 8 = 2,352 张（正常样本） |
| **Test（10%）** | 来自 `rail_mvtec_gt_test`（正常+异常） |

**图像规格**：
- 原始尺寸：6000×900
- Patch 尺寸：900×900
- 输入网络：256×256

---

## ✅ 训练前检查

运行测试脚本，确保数据加载正确：
```bash
python test_rail_dataset.py
```

预期输出：
```
✓ All tests passed!
Train dataset size: 2058
Test dataset size: 147
```

---

## 🎯 训练参数（可调）

### 本地环境
在 `scripts/train_rail_cam1.bat` 中修改：

```batch
set BATCH_SIZE=16      REM 96GB 显存可用 32 或 64
set EPOCHS=100         REM 训练轮数
set LR=0.0001          REM 学习率
set VIEW_ID=1          REM 相机视角（1-8）
set SAMPLE_RATIO=1.0   REM 使用全部数据
```

### 服务器环境
在 `scripts/train_rail_cam1_server.sh` 中修改：

```bash
BATCH_SIZE=16          # 96GB 显存可用 32 或 64
EPOCHS=100             # 训练轮数
LR=0.0001              # 学习率
VIEW_ID=1              # 相机视角（1-8）
SAMPLE_RATIO=0.1       # 使用 10% 数据（约 2940 张）
# SAMPLE_NUM=3000      # 或使用固定数量
```

---

## 📈 训练监控

训练日志保存在：
```
outputs/rail/train_rail_cam1_YYYYMMDD_HHMMSS.log
```

实时查看：
```bash
tail -f outputs/rail/train_rail_cam1_*.log
```

---

## 💾 模型保存

最佳模型保存在：
```
outputs/rail/best_cam1.pth
```

包含：
- `student_rgb`：RGB 学生网络权重
- `student_depth`：Depth 学生网络权重
- `auroc`：最佳 AUROC 分数
- `epoch`：对应的 epoch

---

## 📚 详细文档

- **本地训练指南**：`docs/钢轨训练指南.md`
- **服务器训练指南**：`docs/服务器训练指南.md`
- **串行训练指南**：`docs/串行训练指南.md` ⭐ 新增
- **预加载训练指南**：`docs/预加载训练指南.md`
- **数据集文档**：`docs/code/datasets/rail_dataset.md`
- **完整总结**：`README_RAIL_TRAINING.md`

---

## 🔧 常见问题

### 本地环境

**显存不足**
```bash
# 减小 batch size
set BATCH_SIZE=8
```

**训练不收敛**
```bash
# 降低学习率
set LR=0.00001
```

### 服务器环境

**数据量太大**
```bash
# 减少采样比例
SAMPLE_RATIO=0.05  # 使用 5% 数据
```

**训练速度慢**
```bash
# 增大 batch size
BATCH_SIZE=32

# 增加 workers
--num_workers 8
```

**测试集为空**
```bash
# 确保 view_id 在 1-6 范围内（测试集只有 cam1-6）
VIEW_ID=1
```

---

## 🎓 下一步

完成单视角训练后，可以：
1. **训练所有 8 个视角**：使用 `scripts/train_all_cams_server.sh`
2. **消融实验**：对比不同采样比例、深度归一化方式
3. **进入 Stage B**：多视角融合（PEFT）

详见 `docs/串行训练指南.md` 和 `docs/服务器训练指南.md`

