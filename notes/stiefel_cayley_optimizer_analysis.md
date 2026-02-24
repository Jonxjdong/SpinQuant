# SpinQuant Stiefel 优化器代码分析

> 针对 `train_utils/optimizer.py` L158-L172 的逻辑流程与 Stiefel 流形黎曼梯度投影公式的对比分析。

## Stiefel 流形与切空间

### Stiefel 流形的定义

**Stiefel 流形** $\text{St}(p, n)$ 是所有 $n \times p$（$p \leq n$）的列正交矩阵的集合：

$$\text{St}(p, n) = \{Y \in \mathbb{R}^{n \times p} \mid Y^T Y = I_p\}$$

- 当 $p = n$ 时，$\text{St}(n, n) = O(n)$，即正交群（所有 $n \times n$ 正交矩阵）。
- 当 $p = 1$ 时，$\text{St}(1, n) = S^{n-1}$，即单位球面。

直观理解：Stiefel 流形是嵌入在 $\mathbb{R}^{n \times p}$ 中的一个曲面，上面的每个点都是一个列正交矩阵。"旋转矩阵优化"就是在这个曲面上寻找使损失函数最小的点。

### 切空间的定义

在 Stiefel 流形上某一点 $Y$ 处的**切空间** $T_Y \text{St}(p, n)$ 是所有"沿流形方向"的向量的集合。数学定义为：

$$T_Y \text{St}(p, n) = \{Z \in \mathbb{R}^{n \times p} \mid Y^T Z + Z^T Y = 0\}$$

即切向量 $Z$ 必须满足 $Y^TZ$ 是**反对称矩阵**。

直观理解：

- 在流形上的某一点 $Y$，切空间就是所有"紧贴着曲面"的方向。沿这些方向走一小步，近似仍在流形上。
- 约束 $Y^TZ + Z^TY = 0$ 来源于对正交约束 $Y^TY = I$ 求微分：如果 $Y(t)$ 是流形上的一条曲线，则 $\frac{d}{dt}(Y^TY) = \dot{Y}^TY + Y^T\dot{Y} = 0$。

### 为什么需要切空间？

在欧几里得空间中，梯度下降直接沿梯度反方向走一步。但在 Stiefel 流形上，沿任意方向走一步可能"跳出"流形（不再正交）。因此需要：

1. 将欧几里得梯度**投影到切空间**，得到流形上的合法下降方向
2. 沿该方向更新后，通过**回缩**（retraction）操作将结果拉回流形

SpinQuant 代码中使用的 **Cayley 变换**就是一种回缩方法，它通过反对称矩阵 $W$ 参数化流形上的曲线，天然保证结果的正交性。

---

## 代码逻辑流程详解

### 符号约定

设参数矩阵 `p` 的归一化版本为 `unity`，记为 $X$，其形状为 $(p, n)$，满足 $XX^T = I$（行正交）。代码中 `unity` 实际上就是 $X$，而 `unity.t()` 是 $X^T$，形状为 $(n, p)$。

注意：代码中的约定是 $X$ 是 $p \times n$ 矩阵 ($p \leq n$)，行正交。这等价于标准 Stiefel 流形 $\text{St}(p, n)$ 中以 $X^T$ ($n \times p$) 为元素的写法。

### 逐行分析

```python
V = param_state["momentum_buffer"]        # V: (n, p) —— 动量缓冲
V = momentum * V - g.t()                  # 更新动量: V = μV - ∇f^T，此时 V 是 n×p 矩阵
```

$V$ 是欧几里得梯度（含动量）的转置，形状为 $(n, p)$。

```python
MX = torch.mm(V, unity)          # MX = V · X = V X,  (n,p)×(p,n) = (n,n)
XMX = torch.mm(unity, MX)        # XMX = X · MX = X V X, (p,n)×(n,n) = (p,n)
XXMX = torch.mm(unity.t(), XMX)  # XXMX = X^T · XMX = X^T X V X, (n,p)×(p,n) = (n,n)
```

由于 $XX^T = I_p$，所以：
- $MX = VX$，这是一个 $n \times n$ 矩阵
- $XMX = X \cdot VX = XVX$，是 $p \times n$ 矩阵
- $XXMX = X^T \cdot XVX = X^T X V X$，是 $n \times n$ 矩阵

**注意**：这里 $X^TX \neq I$（因为 $X$ 是 $p \times n$，$p \leq n$，正交关系是 $XX^T = I_p$，而非 $X^TX = I_n$）。

```python
W_hat = MX - 0.5 * XXMX          # W_hat = VX - 0.5 * X^T X V X,  (n,n)
W = W_hat - W_hat.t()            # W = W_hat - W_hat^T，反对称化，(n,n)
```

展开：

$$\hat{W} = VX - \frac{1}{2} X^T X V X$$

$$W = \hat{W} - \hat{W}^T = \left(VX - \frac{1}{2}X^TXVX\right) - \left(X^TV^T - \frac{1}{2}X^TV^TX^TX\right)$$

```python
t = 0.5 * 2 / (matrix_norm_one(W) + episilon)   # 自适应步长
alpha = min(t, lr)                                # 步长上界为 lr
```

```python
p_new = Cayley_loop(unity.t(), W, V, alpha)  # Cayley 迭代更新参数
V_new = torch.mm(W, unity.t())               # 更新动量方向: V_new = W · X^T, (n,n)×(n,p) = (n,p)
```

---

## 与标准 Stiefel 流形黎曼梯度投影公式的对比

### 标准公式

在 Stiefel 流形 $\text{St}(p, n) = \{Y \in \mathbb{R}^{n \times p} \mid Y^T Y = I_p\}$ 上，标准的黎曼梯度投影公式为：

$$\text{grad}_R f(Y) = \nabla f(Y) - Y \cdot \text{sym}(Y^T \nabla f(Y))$$

其中 $\text{sym}(A) = \frac{1}{2}(A + A^T)$。

### 代码中的做法

代码**并不是在直接计算黎曼梯度投影**，而是在为 **Cayley 变换**构建所需的**反对称矩阵** $W$。

这是关键区别！这段代码实现的是 **Wen & Yin (2013)** 的 "Optimization on Stiefel Manifold via Cayley Transform" 方法。在该方法中：

1. **不需要显式计算黎曼梯度**，而是直接构造一个反对称矩阵 $W$，使得沿 $W$ 生成的 Cayley 曲线是流形上的一条下降路径。

2. 该方法的核心思路是：给定欧几里得梯度方向 $G$（代码中通过动量 $V$ 代替），沿流形的切方向可以用一个反对称矩阵来参数化：

   $$Y(\tau) = \left(I + \frac{\tau}{2}W\right)^{-1}\left(I - \frac{\tau}{2}W\right) X^T$$

   这就是 Cayley 变换。`Cayley_loop` 函数就是在迭代求解这个隐式方程。

3. **$W$ 的构造方式**是：

   $$W = \hat{W} - \hat{W}^T$$

   其中 $\hat{W} = GX^T$ 某种修正后的版本（代码中加了 $-\frac{1}{2}X^TXGX^T$ 项来保证切向量在切空间内）。这个构造确保了 $W$ 是反对称的（$W = -W^T$），从而 Cayley 变换的结果仍然在 Stiefel 流形上。

---

## 用户分析检查

### ✅ 正确的部分

1. **步骤1-3的矩阵乘法展开是正确的**：将 `unity` 记为 $X$（行式约定），$MX = VX^T$ 等推导没有问题。

2. **最终观察到代码公式和标准黎曼梯度投影不一致是对的**。

3. **最后的结论"代码并不是在计算标准的 Stiefel 流形投影，而是在构建 Cayley 变换所需要的反对称矩阵"——这个结论完全正确！**

### ⚠️ 需要注意/补充的部分

1. **关于 $X^TX = I$ 的使用**：在步骤3中写到"由于 $X^TX = I$"，这个需要小心。在代码的约定中，`unity` 的形状是 $(p, n)$，$p \leq n$，满足的是 $XX^T = I_p$，**而不是** $X^TX = I_n$。如果 $X$ 指的是 `unity.t()` 即 $(n, p)$ 的话，那么 $X^TX = I_p$ 是对的。需要保持符号一致。

2. **$\hat{W}$ 的简化**：分析中写了 $\hat{W} = VX^T - 0.5 \cdot X \cdot (X^TVX^T) = 0.5VX^T$，这个化简**有问题**。

   用代码中的实际符号（令 `unity` = $U$，形状 $(p, n)$，$UU^T = I_p$）：
   - `MX = V·U`，形状 $(n,p) \times (p,n) = (n, n)$
   - `XMX = U·MX = U·V·U`，形状 $(p, n)$
   - `XXMX = U^T·XMX = U^T·U·V·U`，形状 $(n, n)$
   - $\hat{W} = VU - \frac{1}{2}U^TUVU$

   **不能化简为 $\frac{1}{2}VU$**，因为 $U^TU \neq I_n$（只有 $UU^T = I_p$）。

   所以化简 "$\hat{W} = 0.5VX^T$" 是**不正确的**，除非 $p = n$（方阵情况）。

3. **$W$ 的展开**：由于 $\hat{W}$ 的化简有误，后续得到的 $W = \frac{1}{2}(VX^T - XV^T)$ 也不完全正确（在非方阵情况下）。正确的表达应该是：

   $$W = \left(VU - \frac{1}{2}U^TUVU\right) - \left(U^TV^T - \frac{1}{2}U^TV^TUU^T\right)$$

---

## 总结

| 方面 | 评价 |
|------|------|
| 矩阵乘法展开 | ✅ 正确 |
| 核心结论（不是黎曼投影而是 Cayley 变换构造） | ✅ 完全正确 |
| $X^TX = I$ 的使用 | ⚠️ 需注意 $p < n$ 时只有 $XX^T = I_p$，$X^TX$ 是秩为 $p$ 的投影矩阵 |
| $\hat{W}$ 化简为 $0.5VX^T$ | ❌ 不正确（$U^TU \neq I$ 不能直接消去） |
| $W$ 最终表达式 | ⚠️ 受上一步影响，需要重新推导 |

整体思路和方向是正确的，最关键的洞察——**代码的目的不是计算黎曼梯度投影，而是为 Cayley 变换构建反对称矩阵 $W$**——这一点完全正确。主要问题在于对 $U^TU$ 的处理上，在 $p < n$ 的情况下不能将其当作单位阵来化简。
