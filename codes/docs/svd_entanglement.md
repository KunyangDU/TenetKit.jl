# SVD 与纠缠熵

## 1. 量子态的二分割 (Bipartition)

将系统分为两部分：$A$（站点 $i$ 左侧）和 $B$（站点 $i$ 右侧）。MPS 在 bond $i$ 处的张量 $M$ 可看作线性映射 $M: \mathcal{H}_A \to \mathcal{H}_B$：

$$|\psi\rangle = \sum_{\alpha} \lambda_\alpha |\alpha\rangle_A \otimes |\alpha\rangle_B$$

这就是 **Schmidt 分解**，$\lambda_\alpha \geq 0$ 为 Schmidt 系数，$\{|\alpha\rangle_A\}$ 和 $\{|\alpha\rangle_B\}$ 是左右 Schmidt 基（正交归一）。

## 2. SVD = Schmidt 分解

对 MPS bond 张量做 SVD：

$$M = U \Sigma V^\dagger$$

其中：
- $U$: 正交矩阵，列向量构成 $\mathcal{H}_B$ 的 Schmidt 基 $\{|\alpha\rangle_B\}$
- $V^\dagger$: 正交矩阵，行向量构成 $\mathcal{H}_A$ 的 Schmidt 基 $\{|\alpha\rangle_A\}$
- $\Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_D)$: 对角矩阵，对角元 $\sigma_\alpha$ 就是 Schmidt 系数 $\lambda_\alpha$

展开形式：

$$M_{ab} = \sum_{\alpha=1}^{D} U_{a\alpha} \sigma_\alpha (V^\dagger)_{\alpha b}$$

即 $|\psi\rangle = \sum_\alpha \sigma_\alpha |\alpha\rangle_B \otimes |\alpha\rangle_A$。

## 3. 约化密度矩阵

对 $B$ 求迹得到 $A$ 的约化密度矩阵：

$$\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|) = M^\dagger M = V \Sigma^2 V^\dagger$$

对 $A$ 求迹得到 $B$ 的约化密度矩阵：

$$\rho_B = \text{Tr}_A(|\psi\rangle\langle\psi|) = M M^\dagger = U \Sigma^2 U^\dagger$$

两者有相同的非零本征值谱：$\{\sigma_1^2, \sigma_2^2, \ldots, \sigma_D^2\}$。

## 4. 纠缠熵

von Neumann 纠缠熵定义为：

$$S = -\text{Tr}(\rho_A \ln \rho_A) = -\text{Tr}(\rho_B \ln \rho_B)$$

代入 $\rho_A = V \Sigma^2 V^\dagger$：

$$S = -\sum_{\alpha=1}^{D} \sigma_\alpha^2 \ln \sigma_\alpha^2$$

**结论：SVD 的奇异值平方就是约化密度矩阵的本征值，纠缠熵是奇异值平方的香农熵。**

## 5. 其他纠缠度量

| 度量 | 公式 | 说明 |
|------|------|------|
| Bond 维度 | $D$ | SVD 非零奇异值个数 |
| 纠缠熵 (von Neumann) | $S = -\sum \sigma_\alpha^2 \ln \sigma_\alpha^2$ | 本征谱的香农熵 |
| Rényi 熵 | $S_n = \frac{1}{1-n} \ln \sum \sigma_\alpha^{2n}$ | $n=2$ 为纯度对数的负数 |
| 截断误差 | $\sum_{\alpha > \chi} \sigma_\alpha^2$ | 丢弃的奇异值平方和 |

## 6. 截断与最优近似

Eckart-Young-Mirsky 定理保证：保留前 $\chi$ 个奇异值的截断 SVD 是给定秩 $\chi$ 下的 **Frobenius 范数最优近似**。

截断误差 $\epsilon = \sum_{\alpha > \chi} \sigma_\alpha^2$ 也是对纠缠熵的低估量——丢弃的奇异值越小，损失的纠缠信息越少。这是 DMRG 和 MPS 压缩的核心数学基础。
