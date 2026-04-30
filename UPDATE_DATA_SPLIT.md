# 数据划分与串行训练更新说明

## 更新时间
2026-04-29

## 主要改动

### 1. 数据划分优化 ⭐

**之前**：
- 训练集：所有正常样本
- 测试集：正常 + 异常样本

**现在**：
- **Train（80%）**：正常样本（用于训练）
- **Val（10%）**：正常样本（用于验证和早停）
- **Test（10%）**：正常 + 异常样本（用于最终评估）

**优势**：
- ✅ 避免在测试集上选择模型（防止过拟合）
- ✅ 支持早停（Early Stopping）
- ✅ 更科学的模型评估流程

### 2. 串行训练脚本 ⭐

**新增脚本**：`scripts/train_all_cams_server.sh`

**功能**：
- 支持训练所有 8 个 Cam
- 支持训练指定的 Cam
- 自动记录每个 Cam 的训练时间
- 统计总训练时间

**使用示例**：
```bash
# 训练所有 8 个 Cam
bash scripts/train_all_cams_server.sh

# 训练指定的 Cam
bash scripts/train_all_cams_server.sh 1 2 3

# 只训练 Cam 1
bash scripts/train_all_cams_server.sh 1
```

---

## 修改的文件

### 核心代码

1. **datasets/rail_dataset.py**
   - 添加 `train_val_test_split` 参数
   - 修改 `_scan_files()` 支持 `split="val"`
   - 修改 `_apply_sampling()` 实现 8:1:1 划分
   - 使用固定随机种子保证可复现

2. **train/train_trd_rail.py**
   - 添加 `val_dataset` 和 `val_loader`
   - 修改评估逻辑：在 val 上选择最佳模型
   - 训练结束后在 test 上最终评估
   - 添加 `--train_val_test_split` 参数

3. **eval/eval_utils.py**
   - 修复 `cal_anomaly_map()` 的 bug
   - 修复 `cal_l2dis()` 的 bug
   - 问题：`np.ones([out_size, out_size])` → `np.ones(out_size)`

### 脚本

4. **scripts/train_all_cams_server.sh** ⭐ 新增
   - 串行训练多个 Cam
   - 支持命令行参数指定 Cam
   - 自动统计训练时间

### 文档

5. **docs/串行训练指南.md** ⭐ 新增
   - 数据划分策略说明
   - 串行训练脚本使用指南
   - 参数配置说明
   - 常见问题解答

6. **QUICKSTART.md**
   - 更新数据集统计（8:1:1 划分）
   - 添加串行训练快速开始
   - 更新文档链接

7. **test_data_split.py** ⭐ 新增
   - 测试数据划分功能
   - 验证 8:1:1 比例
   - 检查标签分布

---

## 使用流程

### 1. 测试数据划分（可选）

```bash
python test_data_split.py
```

预期输出：
```
✓ Train/Val only have normal samples
✓ Test has abnormal samples
✓ Train ratio ~80%
✓ Val ratio ~10%
✓ All checks passed!
```

### 2. 训练单个 Cam

```bash
# 使用默认参数（10% 数据，8:1:1 划分）
bash scripts/train_rail_cam1_server_preload.sh
```

### 3. 串行训练多个 Cam

```bash
# 训练所有 8 个 Cam
bash scripts/train_all_cams_server.sh

# 或训练指定的 Cam
bash scripts/train_all_cams_server.sh 1 2 3
```

### 4. 查看结果

```bash
# 查看训练日志
tail -f outputs/rail/*/training.log

# 查看最佳模型
ls outputs/rail/*/best_cam*.pth
```

---

## 训练日志示例

```
Training started at 20260429_120000

[RailDataset] Sampled 2940/29400 images (ratio=10.00%)
[RailDataset] Train split: 2352 images
[RailDataset] Val split: 294 images

[RailDataset] Preloading 2352 images to memory...
Preloading: 100%|████████████| 2352/2352 [02:15<00:00, 17.4 images/s]

Train samples: 16464, Val samples: 2058, Test samples: 147

==================================================
Epoch 1/100
==================================================
Epoch [1] Batch [0/129] Loss_RGB: 2.1234 Loss_Depth: 0.8765
...
Epoch 1 - Avg Loss RGB: 1.8127, Avg Loss Depth: 0.5667

==================================================
Epoch 5/100
==================================================
...
Epoch 5 - Val AUROC: 0.7234
Saved best model to outputs/rail/.../best_cam1.pth

...

Training finished! Best Val AUROC: 0.8567

==================================================
Final evaluation on test set...
==================================================
Final Test AUROC: 0.8234
```

---

## 参数说明

### 新增参数

```bash
--train_val_test_split 0.8 0.1 0.1
```

指定 train/val/test 的划分比例（默认 8:1:1）

### 修改参数

```bash
# 使用 20% 数据
--train_sample_ratio 0.2

# 修改划分比例（7:2:1）
--train_val_test_split 0.7 0.2 0.1
```

---

## 时间估算

### 单个 Cam（10% 数据）

| 阶段 | 时间 |
|-----|------|
| 预加载 | 2-3 分钟 |
| 训练（100 epochs） | 30-50 分钟 |
| **总计** | **35-55 分钟** |

### 8 个 Cam 串行训练

| 数据量 | 单个 Cam | 8 个 Cam |
|-------|---------|---------|
| 10% | 35-55 分钟 | **5-7 小时** |
| 20% | 1-1.5 小时 | **8-12 小时** |
| 50% | 2.5-3.5 小时 | **20-28 小时** |

---

## 常见问题

### 1. Val AUROC 很低（~0.5）？

**正常现象**。Val 只有正常样本，模型难以区分（类似随机猜测）。

**关注 Test AUROC**（有异常样本，才能真正评估性能）。

### 2. 如何修改划分比例？

编辑 `scripts/train_all_cams_server.sh`：

```bash
# 修改为 7:2:1
TRAIN_VAL_TEST_SPLIT="0.7 0.2 0.1"
```

### 3. 如何并行训练？

如果有多张 GPU：

```bash
# GPU 0: 训练 Cam 1-4
CUDA_VISIBLE_DEVICES=0 bash scripts/train_all_cams_server.sh 1 2 3 4 &

# GPU 1: 训练 Cam 5-8
CUDA_VISIBLE_DEVICES=1 bash scripts/train_all_cams_server.sh 5 6 7 8 &
```

---

## 下一步

1. **测试数据划分**：运行 `test_data_split.py`
2. **训练单个 Cam**：验证流程正确
3. **串行训练所有 Cam**：使用 `train_all_cams_server.sh`
4. **评估结果**：对比不同 Cam 的性能

详见 `docs/串行训练指南.md`
