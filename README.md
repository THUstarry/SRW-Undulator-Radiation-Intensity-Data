# SRW波荡器辐射强度代理建模数据集

**SRW Undulator-Radiation Intensity Dataset for Surrogate Modeling**

## 1. 数据集概述 / Dataset overview

本数据集包含1089个SRW多电子波荡器辐射算例。扫描输入为平面波荡器垂直磁场强度 \(B_y\) 和相对于基波共振光子能量的无量纲失谐量 \(\delta_E\)，输出为简化光束线观测平面上的二维Stokes \(S_0\) 强度分布。全部样本均采用线性插值重采样至相同的物理坐标网格。

The dataset contains 1,089 multi-electron SRW simulations. The two inputs are the planar-undulator vertical magnetic-field amplitude \(B_y\) and the dimensionless photon-energy detuning \(\delta_E\) relative to the fundamental resonance. The output is the propagated two-dimensional Stokes \(S_0\) intensity on a common physical grid.

失谐量定义为：

\[
\delta_E =
\frac{E_{\mathrm{photon}}-E_{\mathrm{resonance}}}
     {E_{\mathrm{resonance}}}.
\]

## 2. 文件结构 / Package contents

```text
SRW_Undulator_Radiation_Intensity_Data/
├── srw_intensity_dataset.h5
├── dataset_split_indices.npz
├── README.md
├── metadata.json
├── example_load.py
├── example_train_infer.py
├── LICENSE
└── SHA256SUMS.txt
```

| 文件 / File | 作用 / Purpose |
|---|---|
| `srw_intensity_dataset.h5` | 完整物理数据与统一坐标 / Full physical dataset and common coordinates |
| `dataset_split_indices.npz` | 论文固定数据划分索引 / Fixed split indices used in the paper |
| `README.md` | 中英文数据说明与使用方法 / Bilingual documentation and usage |
| `metadata.json` | 机器可读的物理配置、结构与数据范围 / Machine-readable provenance and schema |
| `example_load.py` | 最小Python读取示例 / Minimal Python loader |
| `example_train_infer.py` | 线性与对数表示下POD-Linear训练、推理及MRE对比示例 / Minimal POD-Linear training, inference, and MRE comparison |
| `LICENSE` | MIT许可条款 / MIT license terms |
| `SHA256SUMS.txt` | 文件完整性校验值 / File-integrity checksums |

## 3. HDF5数据结构 / HDF5 schema

| 数据项 / Dataset | 形状 / Shape | 类型 / Type | 说明 / Description |
|---|---:|---|---|
| `sample_id` | `(1089,)` | `int64` | 稳定样本编号0-1088 / Stable sample ID |
| `parameters` | `(1089, 2)` | `float64` | 列顺序为 `[By_T, detuning_relative]` |
| `By_T` | `(1089,)` | `float64` | 波荡器垂直磁场强度，单位T |
| `detuning_relative` | `(1089,)` | `float64` | 相对共振光子能量失谐量 |
| `intensity` | `(1089,160,160)` | `float32` | 传播后的SRW Stokes \(S_0\)，轴顺序为 `[sample,y,x]` |
| `x_um` | `(160,)` | `float64` | 统一水平物理坐标，单位μm |
| `y_um` | `(160,)` | `float64` | 统一垂直物理坐标，单位μm |
| `n_macro_electrons` | `(1089,)` | `int64` | 每个算例的宏电子数，均为10000 |

## 4. 参数扫描与物理网格 / Parameter scan and physical grid

| 项目 / Item | 设置 / Setting |
|---|---|
| \(B_y\) | 0.60-0.90 T，共33个等间距点 |
| \(\delta_E\) | -0.01-0.01，共33个等间距点 |
| 样本总数 | \(33\times33=1089\) |
| 统一坐标范围 | \(x,y\in[-650,650]\) μm |
| 统一网格 | 160×160 |
| 重采样方式 | 线性插值 |
| 宏电子数 | 每个参数点10000 |

## 5. 固定数据划分 / Fixed data split

参数同时满足以下条件的样本构成内域：

\[
0.65\leq B_y\leq0.85\ {\rm T},
\qquad
-0.008\leq\delta_E\leq0.008.
\]

超出任一内域边界的样本全部归入外推测试集。内域样本使用固定随机种子0进行一次随机排列，再按70%/15%/15%划分为训练集、验证集和内插测试集。该划分由所有代理模型、线性/对数表示以及神经网络训练随机种子共享。

| 集合 / Set | 索引数组 / Index array | 样本数 / Count | 用途 / Role |
|---|---|---:|---|
| 训练集 / Train | `train_indices` | 367 | 拟合与训练 |
| 验证集 / Validation | `validation_indices` | 78 | 提前停止与模型选择 |
| 内插测试集 / Interpolation test | `interp_test_indices` | 80 | 内插独立测试 |
| 外推测试集 / Extrapolation test | `extrap_test_indices` | 564 | 外推独立测试 |

索引值直接对应HDF5各数组的第一维。四个索引数组互不重叠，并且其并集覆盖全部1089个样本。

## 6. 物理算例 / Physical configuration

| 参数 / Parameter | 设置 / Value |
|---|---|
| 波荡器 / Undulator | 平面波荡器；周期长度0.033 m；周期数70 |
| 电子束 / Electron beam | 6 GeV；平均流强0.1 A；相对能散 \(1.1\times10^{-3}\) |
| 电子束RMS | \(\sigma_x=20\) μm，\(\sigma_y=2\) μm，\(\sigma_{x'}=3\) μrad，\(\sigma_{y'}=0.5\) μrad |
| 光束线 / Beamline | 理想薄透镜与透镜后10 m自由空间漂移 |
| 透镜焦距 / Focal length | \(f_x=f_y=10(E_{\mathrm{photon}}/2000\ {\rm eV})^2\) m |
| 模型范围 / Scope | 简化传播系统；不包含实际镜面误差、有限口径或色散元件 |

## 7. 使用方法 / Usage

读取示例依赖Python、NumPy和h5py。运行以下命令可检查数据形状和各集合样本数：

```bash
python example_load.py
```

最小读取示例：

```python
from pathlib import Path

import h5py
import numpy as np

root = Path(".")

with h5py.File(root / "srw_intensity_dataset.h5", "r") as handle:
    parameters = handle["parameters"][:]
    intensity = handle["intensity"][:]
    x_um = handle["x_um"][:]
    y_um = handle["y_um"][:]

with np.load(root / "dataset_split_indices.npz", allow_pickle=False) as split:
    train_indices = split["train_indices"]
    validation_indices = split["validation_indices"]
    interp_test_indices = split["interp_test_indices"]
    extrap_test_indices = split["extrap_test_indices"]

X_train = parameters[train_indices]
Y_train = intensity[train_indices]
```

训练与推理示例另需scikit-learn。在数据集目录运行以下命令，即可分别训练线性与对数表示的POD-Linear模型，并输出两种表示在内插、外推测试集上的平均MRE及差值：

```bash
python example_train_infer.py
```

该示例仅演示固定划分、仅使用训练集拟合预处理参数与POD基、训练和恢复二维物理强度场的流程；不生成SRW数据，也不用于复现其他三类模型或全部论文图表。表中“Log - Linear”为两个MRE百分数之差，负值表示对数表示的MRE较低。

The training/inference example also requires scikit-learn. It fits linear- and log-representation POD-Linear models using the published split, then reports mean MRE on both test sets. It demonstrates the preprocessing and prediction workflow; it does not reproduce the other three models or all paper figures.

## 8. 软件环境 / Software environment

以下为目前在本地SRW项目Python环境中核实的版本，供使用本数据集及示例时参考。将其作为论文原始实验环境记录前，应与当时的运行记录核对。

The versions below were verified in the currently available local SRW Python environment. Confirm them against the original experiment logs before describing them as the exact environment used for the paper results.

| 软件 / Software | 版本 / Version |
|---|---|
| Python | 3.11.14 |
| SRW (`srwpy`) | 4.1.1 |
| PyTorch | 2.11.0.dev20260126+cu128 (CUDA 12.8 build) |
| NumPy | 2.2.6 |
| SciPy | 1.16.2 |
| scikit-learn | 1.7.2 |
| h5py | 3.15.1 |

SRW用于生成原始模拟数据，PyTorch用于论文中的神经网络模型；本目录的POD-Linear示例仅需NumPy、h5py和scikit-learn。

SRW generated the simulation data, and PyTorch was used for the neural-network models in the paper. The bundled POD-Linear example only needs NumPy, h5py, and scikit-learn.


## 9. 完整性检查 / Integrity verification

`SHA256SUMS.txt`记录除其自身外所有发布文件的SHA-256值。下载后可以在数据集目录中使用以下命令逐项校验。

PowerShell：

```powershell
$checks = Get-Content SHA256SUMS.txt
foreach ($line in $checks) {
    $expected, $file = $line -split '\s+', 2
    $actual = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
    "{0}: {1}" -f $file, $(if ($actual -eq $expected) { 'OK' } else { 'MISMATCH' })
}
```

Linux：

```bash
sha256sum -c SHA256SUMS.txt
```

## 10. 许可 / License

本数据集及随附示例代码按照MIT许可证发布，具体条款见[`LICENSE`](LICENSE)。使用时请同时注明数据集来源。

The dataset and bundled example code are released under the MIT License; see [`LICENSE`](LICENSE). Please also cite the dataset when using it.

