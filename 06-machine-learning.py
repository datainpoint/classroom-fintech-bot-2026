#!/usr/bin/env python
# coding: utf-8

# # 臺灣銀行委辦 AI 人才進階訓練
# 
# > 機器學習（二），2026-04-29
# 
# [郭耀仁](https://hahow.in/@tonykuoyj?tr=tonykuoyj) | <yaojenkuo@ntu.edu.tw>

# In[1]:


from pyvizml import GradientDescent
from pyvizml import AdaGrad
from pyvizml import LogitReg
from pyvizml import ClfMetrics


# ## 目錄
# 
# - 分類模型（P.3）
# - 集成模型（P.102）
# - 模型選擇（P.121）
# - 聚類模型（P.163）
# - 時間序列模型（P.168）
# - 隨堂練習（P.184）

# ## 分類模型

# ## 另外一種生成係數向量 $w$ 的演算方法
# 
# - 在機器學習、深度學習中更為廣泛使用的演算方法稱為「梯度遞減」（Gradient descent）。
# - 基本概念是先隨機初始化一組係數向量，以迭代更新該組係數向量，一直到 $J(w)$ 收斂到局部最小值為止。

# ## 為什麼採用梯度遞減
# 
# - 正規方程中必須要透過計算 $X^TX$ 的反矩陣 $(X^TX)^{-1}$ 求解 $w^*$
# - `(n, n)` 反矩陣的計算複雜性最高是 $O(n^3)$，這意味著如果特徵個數變為 2 倍，計算 $(X^TX)^{-1}$ 的時間最多會變為 8 倍。
# - 當特徵矩陣 `n` 很大（約莫是大於 $10^4$），正規方程的計算複雜性問題就會浮現，例如低解析度 $100 \: px \times 100 \: px$ 的灰階圖片。

# ## 梯度遞減如何「有方向性地」更新係數向量
# 
# - 梯度遞減並不是盲目亂槍打鳥地更新係數向量。
# - 依據損失函數 $J(w)$ 關於係數向量 $w$ 的偏微分來決定更新的方向性。
# - 更新幅度則由一個大於零、稱為「學習速率」的常數 $\alpha$ 決定。
# 
# \begin{equation}
# w := w - \alpha \frac{\partial J}{\partial w}
# \end{equation}

# In[2]:


import numpy as np

X0 = np.ones((10, 1))
X1 = np.arange(1, 11).reshape(-1, 1)
w = np.array([5, 6])
X_train = np.concatenate([X0, X1], axis=1)
y_train = np.dot(X_train, w)
print(X_train)
print(y_train)


# ## 從後見之明的視角來看
# 
# 我們知道係數向量 $w^*$ 的組成 $w_0=5$、$w_1=6$
# 
# \begin{equation}
# f(x) = y = 5x_0 + 6x_1 \\
# w^* = \begin{bmatrix} w_0^* \\ w_1^* \end{bmatrix} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}
# \end{equation}

# ## 大海撈針的問題
# 
# - 給定電腦程式一組 $X^{(train)}$ 與 $y^{(train)}$
# - 有無限多組的 $w$ 等著要嘗試。
# - 「梯度遞減」演算方法就是為電腦程式提供了一個尋找解題的方式。
# 
# ![](images/0002.png)

# ## 首先隨機初始化一組 $w$

# In[3]:


np.random.seed(42)
w = np.random.rand(2)
w


# ## 針對這組 $w$ 可以得到一組 $\hat{y}^{(train)}$

# In[4]:


y_hat = np.dot(X_train, w)
y_hat


# ## 針對這組 $\hat{y}^{(train)}$ 可以計算與 $y^{(train)}$ 的均方誤差
# 
# \begin{align}
# \text{MSE}_{train} &= \frac{1}{m}\sum_{i}^{m}{(\hat{y_i}^{(train)} - y^{(train)}_i)^2} \\
#  &= (\hat{y}^{(train)} - y^{(train)})^{T}(\hat{y}^{(train)} - y^{(train)})
# \end{align}

# In[5]:


m = y_train.size
j = ((y_hat - y_train).T.dot(y_hat - y_train)) / m
j


# ## 下一次的該如何更新 $w$ 才能確保離 $w^*$ 更近，讓計算出來的均方誤差會更小一些

# ## 梯度遞減演算方法
# 
# 將目前的 $w_0$ 減去學習速率 $\alpha$ 乘上 $J(w)$ 關於 $w_0$ 的偏微分、將目前的 $w_1$ 減去學習速率 $\alpha$ 乘上 $J(w)$ 關於 $w_1$ 的偏微分。
# 
# \begin{equation}
# w_0 := w_0 - \alpha \frac{\partial J}{\partial w_0}
# \end{equation}
# 
# \begin{equation}
# w_1 := w_1 - \alpha \frac{\partial J}{\partial w_1}
# \end{equation}

# ## 以係數向量的外觀表示
# 
# \begin{equation}
# w := w - \alpha \frac{\partial J}{\partial w}
# \end{equation}

# ## 將 $J(w)$ 關於 $w$ 的偏微分式子展開
# 
# \begin{align}
# \frac{\partial J}{\partial w} &= \frac{1}{m}\frac{\partial}{\partial w}(\parallel y - Xw \parallel^2) \\
# &= \frac{1}{m}\frac{\partial}{\partial w}(Xw - y)^T(Xw-y) \\
# &= \frac{1}{m}\frac{\partial}{\partial w}(w^TX^TXw - w^TX^Ty - y^TXw + y^Ty) \\
# &= \frac{1}{m}\frac{\partial}{\partial w}(w^TX^TXw - (Xw)^Ty - (Xw)^Ty + y^Ty) \\
# &= \frac{1}{m}\frac{\partial}{\partial w}(w^TX^TXw - 2(Xw)^Ty + y^Ty) \\
# &= \frac{1}{m}(2X^TXw - 2X^Ty) \\
# &= \frac{2}{m}(X^TXw - X^Ty) \\
# &= \frac{2}{m}X^T(Xw - y) \\
# &= \frac{2}{m}X^T(\hat{y} - y)
# \end{align}

# ## $J(w)$ 關於 $w$ 的偏微分就是演算方法中所謂的「梯度」（Gradient）
# 
# 在迭代過程中 $w$ 更新的方向性取決於梯度正負號，如果梯度為正，$w$ 會向左更新（減小）；如果梯度為負，$w$ 會向右更新（增大）。
# 
# \begin{equation}
# w := w - \alpha \frac{2}{m}X^T(\hat{y}^{(train)} - y^{(train)})
# \end{equation}

# ## 計算隨機初始化的 $w$ 其梯度為何

# In[6]:


gradients = (2/m) * np.dot(X_train.T, y_hat - y_train)
gradients


# ## 當梯度為負，隨機初始化的 $w$ 會向右更新（增大）
# 
# 離後見之明視角所知的 $w_0 = 5$、$w_1 = 6$ 更加接近，在更新的方向性上是正確的，假設將學習速率設定為 0.001。

# In[7]:


learning_rate = 0.001
-learning_rate * gradients


# ## 經過第一次迭代更新後的 $w$

# In[8]:


w -= learning_rate * gradients
w


# ## 針對更新過一次的 $w$ 可以得到一組 $\hat{y}^{(train)}$

# In[9]:


y_hat = np.dot(X_train, w)
y_hat


# ## 更新過一次的 $w$ 所對應的均方誤差

# In[10]:


j = ((y_hat - y_train).T.dot(y_hat - y_train)) / m
j


# ## 觀察運用「梯度遞減」演算方法
# 
# - 透過計算損失函數關於係數向量的梯度決定更新的**方向性**。
# - 透過學習速率決定更新的**幅度**。
# - 在迭代進行一次之後，係數向右更新（增大）離的 $w^*$ 更接近了些、均方誤差也下降了些。

# ## 自行定義梯度遞減類別 GradientDescent

# ```python
# class GradientDescent:
#     """
#     This class defines the vanilla gradient descent algorithm for linear regression.
#     Args:
#         fit_intercept (bool): Whether to add intercept for this model.
#     """
#     def __init__(self, fit_intercept=True):
#         self._fit_intercept = fit_intercept
# ```

# ```python
#     def find_gradient(self):
#         """
#         This function returns the gradient given certain model weights.
#         """
#         y_hat = np.dot(self._X_train, self._w)
#         gradient = (2/self._m) * np.dot(self._X_train.T, y_hat - self._y_train)
#         return gradient
#     def mean_squared_error(self):
#         """
#         This function returns the mean squared error given certain model weights.
#         """
#         y_hat = np.dot(self._X_train, self._w)
#         mse = ((y_hat - self._y_train).T.dot(y_hat - self._y_train)) / self._m
#         return mse
# ```

# ```python
#     def fit(self, X_train, y_train, epochs=10000, learning_rate=0.001):
#         """
#         This function uses vanilla gradient descent to solve for weights of this model.
#         Args:
#             X_train (ndarray): 2d-array for feature matrix of training data.
#             y_train (ndarray): 1d-array for target vector of training data.
#             epochs (int): The number of iterations to update the model weights.
#             learning_rate (float): The learning rate of gradient descent.
#         """
#         self._X_train = X_train.copy()
#         self._y_train = y_train.copy()
#         self._m = self._X_train.shape[0]
#         if self._fit_intercept:
#             X0 = np.ones((self._m, 1), dtype=float)
#             self._X_train = np.concatenate([X0, self._X_train], axis=1)
#         n = self._X_train.shape[1]
#         self._w = np.random.rand(n)
#         n_prints = 10
#         print_iter = epochs // n_prints
#         w_history = dict()
#         for i in range(epochs):
#             current_w = self._w.copy()
#             w_history[i] = current_w
#             mse = self.mean_squared_error()
#             gradient = self.find_gradient()
#             if i % print_iter == 0:
#                 print("epoch: {:6} - loss: {:.6f}".format(i, mse))
#             self._w -= learning_rate*gradient
#         w_ravel = self._w.copy().ravel()
#         self.intercept_ = w_ravel[0]
#         self.coef_ = w_ravel[1:]
#         self._w_history = w_history
# ```

# ```python
#     def predict(self, X_test):
#         """
#         This function returns predicted values with weights of this model.
#         Args:
#             X_test (ndarray): 2d-array for feature matrix of test data.
#         """
#         self._X_test = X_test
#         m = self._X_test.shape[0]
#         if self._fit_intercept:
#             X0 = np.ones((m, 1), dtype=float)
#             self._X_test = np.concatenate([X0, self._X_test], axis=1)
#         y_pred = np.dot(self._X_test, self._w)
#         return y_pred
# ```

# In[11]:


h = GradientDescent(fit_intercept=False)
h.fit(X_train, y_train, epochs=20000, learning_rate=0.001)


# In[12]:


print(h.intercept_) # 截距項
print(h.coef_)      # 係數項


# ## 將自行定義的梯度遞減預測器類別應用在真實資料

# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split

player_stats = pd.read_csv("https://raw.githubusercontent.com/yaojenkuo/ml-newbies/master/player_stats.csv")
X = player_stats['heightMeters'].values.reshape(-1, 1)
y = player_stats['weightKilograms'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
h = GradientDescent()
h.fit(X_train, y_train, epochs=300000, learning_rate=0.01)


# In[14]:


print(h.intercept_) # 截距項
print(h.coef_)      # 係數項


# In[15]:


# 預測
y_hat = h.predict(X_valid)
y_hat[:5]


# ## 回顧梯度遞減的核心概念
# 
# \begin{equation}
# w := w - \alpha \frac{\partial J}{\partial w}
# \end{equation}
# 
# - $w$ 的更新依據有兩個：學習速率 $\alpha$ 與梯度 $\frac{\partial J}{\partial w}$
# - 學習速率使用一個事先決定的常數。

# ## 不效率的最適化
# 
# - 使用固定的學習速率。
# - 只考慮單下的梯度。
# - 像是用同一套裝備與配速去面對距離不同的路跑賽事。

# ## 以 [Kaggle](https://www.kaggle.com/) 網站所下載回來的[艾姆斯房價](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)資料為例

# In[16]:


from sklearn.linear_model import LinearRegression

# 以其中的 `GrLivArea` 作為特徵矩陣來預測目標向量 `SalePrice`
train = pd.read_csv("https://raw.githubusercontent.com/datainpoint/classroom-ml-from-scratch/main/data/house-prices/train.csv")
X = train['GrLivArea'].values.reshape(-1, 1)
y = train['SalePrice'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.intercept_)
print(lr.coef_)


# ## 使用自行定義的 `GradientDescent` 類別，會發現不論怎麼調整學習速率、增加訓練的迭代次數，$w$ 都離理想值距離甚遠

# In[17]:


h = GradientDescent()
h.fit(X_train, y_train, epochs=500000, learning_rate=1e-7) # 無法使用更大的學習速率，誤差會高到發生溢位


# In[18]:


print(h.intercept_)
print(h.coef_)


# In[19]:


import matplotlib.pyplot as plt

def plot_contour(X_train, y_train, w_history, w_0_min, w_0_max, w_1_min, w_1_max, w_0_star, w_1_star):
    m = X_train.shape[0]
    X0 = np.ones((m, 1), dtype=float)
    X_train = np.concatenate([X0, X_train], axis=1)
    resolution = 100
    W_0, W_1 = np.meshgrid(np.linspace(w_0_min, w_0_max, resolution), np.linspace(w_1_min, w_1_max, resolution))
    Z = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            w = np.array([W_0[i, j], W_1[i, j]])
            y_hat = np.dot(X_train, w)
            mse = ((y_hat - y_train).T.dot(y_hat - y_train)) / m
            Z[i, j] = mse
    epochs = len(w_history)
    w_0_history = []
    w_1_history = []
    for i in range(epochs):
        w_0_history.append(w_history[i][0])
        w_1_history.append(w_history[i][1])
    fig, ax = plt.subplots()
    CS = ax.contour(W_0, W_1, Z)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.plot(w_0_history, w_1_history, "-", color="blue")
    ax.scatter(w_0_star, w_1_star, marker="*", color="red")
    ax.set_xlabel("$w_0$")
    ax.set_ylabel("$w_1$", rotation=0)
    plt.show()


# In[20]:


w_history = h._w_history
plot_contour(X_train, y_train, w_history, -5000, 35000, -10, 200, lr.intercept_, lr.coef_[0])


# ## 搭配兩種技法來增加效率
# 
# 1. 特徵矩陣的標準化。
# 2. 進階的梯度遞減演算方法。

# ## 特徵矩陣的標準化：最小最大標準化（Min-max scaler）
# 
# 標準化後得到的 $w^{(scaled)}$ 要再記得實施「逆」轉換。
# 
# \begin{align}
# \hat{y} &= X^{(scaled)} w^{(scaled)} \\
# &= w_0^{(scaled)}x_0 + \sum_i w_i^{(scaled)} x_i^{(scaled)} \\
# &= w_0^{(scaled)} + \sum_i w_i^{(scaled)} \frac{x_i - x_i^{(min)}}{x_i^{(max)} - x_i^{(min)}}
# \end{align}
# 
# \begin{align}
# w_0 &= w_0^{(scaled)} - \sum_{i=1} w_i^{(scaled)} \frac{x_i^{(min)}}{x_i^{(max)} - x_i^{(min)}} \\
# w_i &= \sum_{i=1} \frac{w_i^{(scaled)}}{x_i^{(max)} - x_i^{(min)}}
# \end{align}

# In[21]:


from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_scaled = mms.fit_transform(X)
y = train['SalePrice'].values
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.33, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.intercept_) # 截距項
print(lr.coef_)      # 係數項


# In[22]:


h = GradientDescent()
h.fit(X_train, y_train, epochs=100000, learning_rate=0.01)
print(h.intercept_) # 截距項
print(h.coef_)      # 係數項


# ## 依照「逆」標準化回推 $w$

# In[23]:


intercept_rescaled = h.intercept_ - (h.coef_ * mms.data_min_ / mms.data_range_)
coef_rescaled = h.coef_ / mms.data_range_
print(intercept_rescaled) # 截距項
print(coef_rescaled)      # 係數項


# ## 特徵矩陣的標準化：常態標準化（Standard scaler）
# 
# 標準化後得到的 $w^{(scaled)}$ 要再記得實施「逆」轉換。
# 
# \begin{align}
# \hat{y} &= X^{(scaled)} w^{(scaled)} \\
# &= w_0^{(scaled)}x_0 + \sum_i w_i^{(scaled)} x_i^{(scaled)} \\
# &= w_0^{(scaled)} + \sum_i w_i^{(scaled)} \frac{x_i - \mu_{x_i}}{\sigma_{x_i}}
# \end{align}
# 
# \begin{align}
# w_0 &= w_0^{(scaled)} - \sum_{i=1} w_i^{(scaled)} \frac{\mu_{x_i}}{\sigma_{x_i}} \\
# w_i &= \sum_{i=1} \frac{w_i^{(scaled)}}{\sigma_{x_i}}
# \end{align}

# In[24]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_scaled = ss.fit_transform(X)
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.33, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.intercept_) # 截距項
print(lr.coef_)      # 係數項


# In[25]:


h = GradientDescent()
h.fit(X_train, y_train, epochs=10000, learning_rate=0.001)
print(h.intercept_) # 截距項
print(h.coef_)      # 係數項


# ## 依照「逆」標準化回推 $w$

# In[26]:


intercept_rescaled = h.intercept_ - h.coef_ * ss.mean_ / ss.scale_
coef_rescaled = h.coef_ / ss.scale_
print(intercept_rescaled) # 截距項
print(coef_rescaled)      # 係數項


# ## 進階的梯度遞減
# 
# - 處於蓬勃發展的階段。
# - 已經廣泛被資料科學家、機器學習工程師應用的有 Momentum、AdaGrad(Adaptive Gradient Descent)、RMSprop(Root mean square propagation)與 Adam(Adaptive moment estimation)。

# ## 從學習速率與梯度這兩方面著手調整
# 
# - 引進調適的學習速率（Adaptive methods），如果距離 $J(w)$ 低點遠就用大的學習速率、反之距離近就用小的學習速率
# - 記錄從訓練開始的梯度量值，藉由過去已實現的梯度來判斷和 $J(w)$ 低點的相對位置，如果歷史梯度都很大，表示離低點遠，如果歷史梯度都很小，表示離低點近。

# ## 以 AdaGrad 為例，將原本梯度遞減的式子改寫
# 
# \begin{equation}
# ssg = \sum^{t-1} (\frac{\partial J}{\partial w})^2
# \end{equation}
# 
# \begin{equation}
# w := w -\alpha \frac{1}{\epsilon + \sqrt{ssg}} \frac{\partial J}{\partial w}
# \end{equation}
# 
# \begin{equation}
# where \; \epsilon = 10^{-6}
# \end{equation}

# ## 記錄歷史梯度的平方和來調適學習速率
# 
# - 當歷史梯度的平方和愈大，會調降學習速率。
# - 當歷史梯度的平方和愈小，會調升學習速率。
# - $\epsilon$ 會設定一個極小值（例如 `1e-06`）避免分母為零的情況發生。

# ## 自定義一個 `AdaGrad` 類別繼承 `GradientDescent` 類別並改寫其 `fit()` 方法

# ```python
# class AdaGrad(GradientDescent):
#     """
#     This class defines the Adaptive Gradient Descent algorithm for linear regression.
#     """
#     def fit(self, X_train, y_train, epochs=10000, learning_rate=0.01, epsilon=1e-06):
#         self._X_train = X_train.copy()
#         self._y_train = y_train.copy()
#         self._m = self._X_train.shape[0]
#         if self._fit_intercept:
#             X0 = np.ones((self._m, 1), dtype=float)
#             self._X_train = np.concatenate([X0, self._X_train], axis=1)
#         n = self._X_train.shape[1]
#         self._w = np.random.rand(n)
#         # 初始化 ssg
#         ssg = np.zeros(n, dtype=float)
#         n_prints = 10
#         print_iter = epochs // n_prints
#         w_history = dict()
#         for i in range(epochs):
#             current_w = self._w.copy()
#             w_history[i] = current_w
#             mse = self.mean_squared_error()
#             gradient = self.find_gradient()
#             ssg += gradient**2
#             ada_grad = gradient / (epsilon + ssg**0.5)
#             if i % print_iter == 0:
#                 print("epoch: {:6} - loss: {:.6f}".format(i, mse))
#             # 以 adaptive gradient 更新 w
#             self._w -= learning_rate*ada_grad
#         w_ravel = self._w.copy().ravel()
#         self.intercept_ = w_ravel[0]
#         self.coef_ = w_ravel[1:]
#         self._w_history = w_history
# ```

# In[27]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
h = AdaGrad()
h.fit(X_train, y_train, epochs=500000, learning_rate=100)
print(h.intercept_) # 截距項
print(h.coef_)      # 係數項


# In[28]:


w_history = h._w_history
plot_contour(X_train, y_train, w_history, -5000, 35000, -10, 200, lr.intercept_, lr.coef_[0])


# ## 在預測類別的前一步是預測機率
# 
# \begin{equation}
# \hat{y} = 1, \quad \text{if} \: \hat{p}(y=1|X; w) \geq \hat{p}(y=0|X; w) \\
# \hat{y} = 0, \quad \text{if} \: \hat{p}(y=1|X; w) < \hat{p}(y=0|X; w)
# \end{equation}

# ## 承先啟後的橋樑
# 
# - 羅吉斯迴歸（Logistic Regression）分類器能夠協助我們由數值預測過渡至類別預測的任務。
# - 羅吉斯迴歸（Logistic Regression）分類器能夠協助我們理解深度學習的基礎理論。
# - 欲得到類別預測 $\hat{y}$ 必須先得到類別預測機率 $\hat{p}$。

# ## Sigmoid 函數
# 
# \begin{equation}
# \sigma(x) = \frac{1}{1 + e^{-x}} \\
# \hat{p} = \sigma(Xw) = \frac{1}{1 + e^{-Xw}}
# \end{equation}

# In[29]:


import numpy as np

def sigmoid(x):
    return(1 / (1 + np.exp(-x)))


# In[30]:


def plot_sigmoid():
    x = np.linspace(-6, 6, 100)
    y = sigmoid(x)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x, y)
    ax.axvline(0, color = 'black')
    ax.axhline(y = 0, ls = ':', color = 'k', alpha = 0.5)
    ax.axhline(y = 0.5, ls = ':', color = 'k', alpha = 0.5)
    ax.axhline(y = 1, ls = ':', color = 'k', alpha = 0.5)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Sigmoid function")
    plt.show()


# In[31]:


plot_sigmoid()


# ## 將迴歸模型的輸出 $Xw$ 映射至 $[0, 1]$ 之間就能獲得 $\hat{p}$
# 
# - 依據門檻值獲得 $\hat{y}$
# - 此處門檻值以常見的 50% 表示。
# 
# \begin{equation}
# \hat{y} = 1, \quad \text{if} \: \hat{p} \geq 0.5 \\
# \hat{y} = 0, \quad \text{if} \: \hat{p} < 0.5
# \end{equation}

# ## 將門檻值比較表示為階躍函式（Step function）
# 
# \begin{align}
# \hat{y} &= h(X; w) \\
# &= \chi(\sigma(Xw))
# \end{align}
# 
# \begin{equation}
# \sigma(x) = \frac{1}{1 + e^{-x}} \\
# \chi(z) = 1, \quad \text{if} \: z \geq 0.5 \\
# \chi(z) = 0, \quad \text{if} \: z < 0.5
# \end{equation}

# ## 定義評估
# 
# 評估 $h$ 的方法是計算 $\hat{y}^{(train)}$ 與 $y^{(train)}$ 之間的誤分類數，誤分類數愈低，分類器的表現愈好。
# 
# \begin{align}
# \operatorname*{arg\,min}_w \; J(w) = \sum_i n(E_i) \\ \text{ where } E_i \; \text{represents the occurrence of } y_i \neq \hat{y_i}
# \end{align}

# ## 使用交叉熵（Cross-entropy）作為損失函數 $J(w)$
# 
# 為了書寫方便，我們省略訓練資料的註記$(train)$。
# 
# \begin{equation}
# J(w) = -\frac{1}{m}log(\sigma(Xw)), \quad \text{if} \: y = 1 \\
# J(w) = -\frac{1}{m}log(1-\sigma(Xw)), \quad \text{if} \: y = 0
# \end{equation}

# ## 交叉熵巧妙之處：讓誤分類的損失趨近無限大
# 
# - 當真實的類別 $y$ 為 1，$\sigma(Xw)$ 若離 0 比較近，預測為類別 0 的機率較高。
# - 當真實的類別 $y$ 為 0，$\sigma(Xw)$ 若離 1 比較近，預測為類別 1 的機率較高。

# In[32]:


def plot_cross_entropy():
    epsilon = 1e-5
    h = np.linspace(epsilon, 1-epsilon) # 利用微小值 epsilon 避免 log(0) 的錯誤
    y1 = -np.log(h)
    y2 = -np.log(1 - h)
    fig, ax = plt.subplots(1, 2, figsize = (8, 4))
    ax[0].plot(h, y1)
    ax[0].set_title("$y=1$\n$-\log(\sigma(Xw))$")
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels([0, 1])
    ax[0].set_xlabel("$\sigma(Xw)$")
    ax[1].plot(h, y2)
    ax[1].set_title("$y=0$\n$-\log(1-\sigma(Xw))$")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels([0, 1])
    ax[1].set_xlabel("$\sigma(Xw)$")
    plt.tight_layout()
    plt.show()


# In[33]:


plot_cross_entropy()


# ## 將 $y$ 與 $1-y$ 加入 $J(w)$ 把兩個情境（$y=0$ 或 $y=1$）合而為一
# 
# 當 $y=1$ 時，$J(w)$ 只剩下前項；當 $y=0$ 時，$J(w)$ 只剩下後項。
# 
# \begin{equation}
# J(w) = \frac{1}{m}(-ylog(\sigma(Xw)) - (1-y)log(1-\sigma(Xw)))
# \end{equation}

# ## 運用梯度遞減找到係數向量 $w^*$
# 
# \begin{equation}
# w := w - \alpha \frac{\partial J}{\partial w}
# \end{equation}

# ## 求解 $J(w)$ 關於 $w$ 的偏微分得具備三個先修知識
# 
# 1. 連鎖法則（Chain rule）。
# 2. $e^{x}$ 關於 $x$ 的微分。
# 3. $log(x)$ 關於 $x$ 的微分。

# ## $J(w)$ 是一個由多個不同函數複合而成的損失函數
# 
# - 先是 Sigmoid 函數 $\sigma$
# - 再來是 $log$ 函數。
# - 欲求解複合函式偏微分得仰賴連鎖法則。
# 
# \begin{align}
# (f\circ g)(x) &= f(g(x)) \\
# (f\circ g)'(x) &= f'(g(x))g'(x)
# \end{align}

# ## $e^{x}$ 關於 $x$ 的微分
# 
# \begin{equation}
# \frac{d}{dx}e^{x} = e^{x}
# \end{equation}

# ## $log(x)$ 關於 $x$ 的微分
# 
# \begin{equation}
# \frac{d}{dx}log(x) = \frac{1}{x}
# \end{equation}

# ## 推導 $J(w)$ 關於 $w$ 的偏微分
# 
# \begin{align}
# \frac{\partial}{\partial w}J &= \frac{\partial}{\partial w} (-ylog(\sigma(Xw)) - (1-y)log(1-\sigma(Xw))) \\
# &= -y\frac{\partial}{\partial w}log(\sigma(Xw)) - (1-y)\frac{\partial}{\partial w}(log(1-\sigma(Xw)))
# \end{align}

# ## 首先計算 $log(\sigma(Xw))$ 關於 $w$ 的微分
# 
# \begin{align}
# \frac{\partial}{\partial w}log(\sigma(Xw)) &= \frac{\partial}{\partial w}log(\sigma(Xw)) \cdot \frac{\partial}{\partial w}(\sigma(Xw)) \\
# &= \frac{1}{\sigma(Xw)} \cdot \sigma'(Xw) \cdot \frac{\partial}{\partial w}Xw \\
# &= \frac{1}{\sigma(Xw)} \cdot \sigma'(Xw) \cdot X
# \end{align}

# ## 再計算 $log(1-\sigma(Xw))$ 關於 $w$ 的微分
# 
# \begin{align}
# \frac{\partial}{\partial w}log(1-\sigma(Xw)) &= \frac{\partial}{\partial w}log(1-\sigma(Xw)) \cdot \frac{\partial}{\partial w}(1-\sigma(Xw)) \\
# &=\frac{1}{1-\sigma(Xw)} \cdot (-\sigma'(Xw) \cdot \frac{\partial}{\partial w}Xw) \\
# &=\frac{1}{1-\sigma(Xw)} \cdot (-\sigma'(Xw) \cdot X)
# \end{align}

# ## 兩個部分都得先計算 $\sigma'(Xw)$ 也就是 Sigmoid 函數關於 $w$ 的微分，才能繼續推導
# 
# \begin{align}
# \sigma'(Xw) &= \frac{\partial}{\partial w} \frac{1}{1 + e^{-Xw}} = \frac{\partial}{\partial w} (1 + e^{-Xw})^{-1} \\
# &= \frac{-\frac{\partial}{\partial w}(1 + e^{-Xw})}{(1 + e^{-Xw})^2}
# \end{align}

# ## 分子部分我們先推導 $e^{-x}$ 關於 $x$ 的微分
# 
# \begin{equation}
# \frac{d}{dx}e^{-x} = \frac{d}{dx}\frac{1}{e^x} = \frac{-\frac{d}{dx} e^x}{(e^x)^2} = \frac{-e^x}{(e^x)^2} = \frac{-1}{e^x} = -e^{-x}
# \end{equation}

# ## 於是 $\sigma'(Xw)$ 可以寫成
# 
# \begin{align}
# \sigma'(Xw) &= \frac{-\frac{\partial}{\partial w}e^{-Xw}}{(1 + e^{-Xw})^2} = \frac{e^{-Xw}}{(1 + e^{-Xw})^2} \\
# &= \frac{e^{-Xw}}{(1 + e^{-Xw}) \cdot (1 + e^{-Xw})}
# \end{align}

# ## 這裡的推導有些狡猾，需要在分子設計一個 `+1-1`
# 
# \begin{align}
# \sigma'(Xw) &= \frac{e^{-Xw}}{(1 + e^{-Xw}) \cdot (1 + e^{-Xw})} \\
# &= \frac{1}{1 + e^{-Xw}} \cdot \frac{e^{-Xw} + 1 - 1}{1 + e^{-Xw}} = \frac{1}{1 + e^{-Xw}} \cdot ( \frac{1 + e^{-Xw}}{1 + e^{-Xw}} - \frac{1}{1 + e^{-Xw}}) \\
# &=\frac{1}{1 + e^{-Xw}} \cdot ( 1 - \frac{1}{1 + e^{-Xw}}) \\
# &=\sigma(Xw)(1-\sigma(Xw))
# \end{align}

# ## 推導出 $\sigma'(Xw)$，再回去計算未完的兩個部分
# 
# \begin{align}
# \frac{\partial}{\partial w}log(\sigma(Xw)) &= \frac{1}{\sigma(Xw)} \cdot \sigma'(Xw) \cdot X \\
# &= \frac{1}{\sigma(Xw)}\sigma(Xw)(1-\sigma(Xw))X \\
# &= (1-\sigma(Xw))X
# \end{align}
# 
# \begin{align}
# \frac{\partial}{\partial w}log(1-\sigma(Xw)) &= \frac{1}{1-\sigma(Xw)} \cdot (-\sigma'(Xw)) \cdot X\\
# &=\frac{1}{1-\sigma(Xw)}(-(\sigma(Xw)(1-\sigma(Xw)))X) \\
# &=-\sigma(Xw)X
# \end{align}

# ## 最後回到 $J(w)$ 關於 $w$ 的偏微分
# 
# \begin{align}
# \frac{\partial J}{\partial w} &= \frac{1}{m}(-y(1-\sigma(Xw))X - (1-y)(-\sigma(Xw)X)) \\
# &=\frac{1}{m}(-X^Ty + y\sigma(Xw)X + X^T\sigma(Xw) - y\sigma(Xw)X) \\
# &=\frac{1}{m}(-X^Ty + X^T\sigma(Xw)) \\
# &=\frac{1}{m}(X^T(\sigma(Xw) - y))
# \end{align}

# ## 梯度推導完畢
# 
# 在迭代過程中 $w$ 更新的方向性取決於梯度正負號，如果梯度為正，$w$ 會向左更新（減小）；如果梯度為負，$w$ 會向右更新（增大）。
# 
# \begin{equation}
# w := w - \alpha \frac{1}{m}(X^T(\sigma(Xw) - y)) \\
# w := w - \alpha \frac{1}{m}(X^T(\sigma(\hat{y}) - y))
# \end{equation}

# ## 自訂羅吉斯迴歸類別 LogitReg

# ```python
# class LogitReg:
#     """
#     This class defines the vanilla descent algorithm for logistic regression.
#     Args:
#         fit_intercept (bool): Whether to add intercept for this model.
#     """
#     def __init__(self, fit_intercept=True):
#         self._fit_intercept = fit_intercept
# ```

# ```python
#     def sigmoid(self, X):
#         """
#         This function returns the Sigmoid output as a probability given certain model weights.
#         """
#         X_w = np.dot(X, self._w)
#         p_hat = 1 / (1 + np.exp(-X_w))
#         return p_hat
#     def find_gradient(self):
#         """
#         This function returns the gradient given certain model weights.
#         """
#         m = self._m
#         p_hat = self.sigmoid(self._X_train)
#         X_train_T = np.transpose(self._X_train)
#         gradient = (1/m) * np.dot(X_train_T, p_hat - self._y_train)
#         return gradient
# ```

# ```python
#     def cross_entropy(self, epsilon=1e-06):
#         """
#         This function returns the cross entropy given certain model weights.
#         """
#         m = self._m
#         p_hat = self.sigmoid(self._X_train)
#         cost_y1 = -np.dot(self._y_train, np.log(p_hat + epsilon))
#         cost_y0 = -np.dot(1 - self._y_train, np.log(1 - p_hat + epsilon))
#         cross_entropy = (cost_y1 + cost_y0) / m
#         return cross_entropy
# ```

# ```python
#     def fit(self, X_train, y_train, epochs=10000, learning_rate=0.001):
#         """
#         This function uses vanilla gradient descent to solve for weights of this model.
#         Args:
#             X_train (ndarray): 2d-array for feature matrix of training data.
#             y_train (ndarray): 1d-array for target vector of training data.
#             epochs (int): The number of iterations to update the model weights.
#             learning_rate (float): The learning rate of gradient descent.
#         """
#         self._X_train = X_train.copy()
#         self._y_train = y_train.copy()
#         m = self._X_train.shape[0]
#         self._m = m
#         if self._fit_intercept:
#             X0 = np.ones((self._m, 1), dtype=float)
#             self._X_train = np.concatenate([X0, self._X_train], axis=1)
#         n = self._X_train.shape[1]
#         self._w = np.random.rand(n)
#         n_prints = 10
#         print_iter = epochs // n_prints
#         for i in range(epochs):
#             cross_entropy = self.cross_entropy()
#             gradient = self.find_gradient()
#             if i % print_iter == 0:
#                 print("epoch: {:6} - loss: {:.6f}".format(i, cross_entropy))
#             self._w -= learning_rate*gradient
#         w_ravel = self._w.ravel().copy()
#         self.intercept_ = w_ravel[0]
#         self.coef_ = w_ravel[1:].reshape(1, -1)
# ```

# ```python
#     def predict_proba(self, X_test):
#         """
#         This function returns predicted probability with weights of this model.
#         Args:
#             X_test (ndarray): 2d-array for feature matrix of test data.
#         """
#         m = X_test.shape[0]
#         if self._fit_intercept:
#             X0 = np.ones((m, 1), dtype=float)
#             self._X_test = np.concatenate([X0, X_test], axis=1)
#         p_hat_1 = self.sigmoid(self._X_test).reshape(-1, 1)
#         p_hat_0 = 1 - p_hat_1
#         proba = np.concatenate([p_hat_0, p_hat_1], axis=1)
#         return proba
#     def predict(self, X_test):
#         """
#         This function returns predicted label with weights of this model.
#         Args:
#             X_test (ndarray): 2d-array for feature matrix of test data.
#         """
#         proba = self.predict_proba(X_test)
#         y_pred = np.argmax(proba, axis=1)
#         return y_pred
# ```

# In[34]:


X = player_stats[['apg', 'rpg']].values
pos = player_stats['pos'].values
position_dictionary = {
    0: "G",
    1: "F"
}
y = np.array([0 if p[0] == 'G' else 1 for p in pos])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
h = LogitReg()
h.fit(X_train, y_train, 100000, 0.01)


# In[35]:


## 預測機率
p_hat = h.predict_proba(X_valid)
p_hat[:5]


# In[36]:


## 預測類別
y_hat = h.predict(X_valid)
y_hat_label = [position_dictionary[y] for y in y_hat]
print(y_hat[:5])
print(y_hat_label[:5])


# ## 原始資料中球員的鋒衛位置不只分作前鋒（Forward, F）與後衛（Guard, G）亦有中鋒（Center, C），以及能夠勝任兩個位置的搖擺人（F-G、G-F）等

# In[37]:


pos = player_stats['pos'].values
print(np.unique(pos))
print(np.unique(pos).size)


# ## One versus rest(all)
# 
# 訓練 7 個羅吉斯迴歸分類器，每個鋒衛位置一個，輸出預測機率，再以 `np.argmax()` 函數決定預測類別。
# 
# \begin{align}
# \hat{p}_{C} = \hat{p}(y=0|X; w) = 1 - \hat{p}(y \neq 0|X; w) \\
# \hat{p}_{C-F} = \hat{p}(y=1|X; w) = 1 - \hat{p}(y \neq 1|X; w)  \\
# \hat{p}_{F} = \hat{p}(y=2|X; w) = 1 - \hat{p}(y \neq 2|X; w)  \\
# \hat{p}_{F-C} = \hat{p}(y=3|X; w) = 1 - \hat{p}(y \neq 3|X; w)  \\
# \hat{p}_{F-G} = \hat{p}(y=4|X; w) = 1 - \hat{p}(y \neq 4|X; w)  \\
# \hat{p}_{G} = \hat{p}(y=5|X; w) = 1 - \hat{p}(y \neq 5|X; w)  \\
# \hat{p}_{G-F} = \hat{p}(y=6|X; w) = 1 - \hat{p}(y \neq 6|X; w)  \\
# \hat{p} = \operatorname*{arg\,max}_\hat{p} (\hat{p}_{C}, \hat{p}_{C-F}, \hat{p}_{F}, \hat{p}_{F-C}, \hat{p}_{F-G}, \hat{p}_{G}, \hat{p}_{G-F})
# \end{align}

# In[38]:


unique_pos = player_stats['pos'].unique()
position_dictionary = {i: p for i, p in enumerate(unique_pos)}
position_dictionary_reversed = {v: k for k, v in position_dictionary.items()}
pos_multiple = player_stats['pos'].map(position_dictionary_reversed)
print(position_dictionary)
print(position_dictionary_reversed)
print(np.unique(pos_multiple))


# ## 使用 Scikit-Learn 的 `LogisticRegression` 類別
# 
# - 在初始化時加入參數 `multi_class='ovr'` 就能面對多元分類問題。
# - `predict_proba()` 方法輸出的陣列外觀為 `(m, n)`
# - 第 0 欄是預測為類別 0（C）的機率 $\hat{p}(y=0|X; w)$、第 6 欄是預測為類別 6（G-F）的機率 $\hat{p}(y=6|X; w)$

# In[39]:


from sklearn.linear_model import LogisticRegression

X = player_stats[['apg', 'rpg']].values
y = pos_multiple
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
h = LogisticRegression(C=1e6, multi_class='ovr')
h.fit(X_train, y_train)
p_hat = h.predict_proba(X_valid)
p_hat[:5]


# ## 應用 `np.argmax()` 函數回傳最大的欄位數，就能夠得到 $\hat{y}$

# In[40]:


y_hat = np.argmax(p_hat, axis=1)
y_hat[:5]


# ## 將整數對應回鋒衛位置的文字外觀

# In[41]:


y_hat_label = [position_dictionary[i] for i in y_hat]
y_hat_label[:5]


# ## 兩種表示類別向量的形式
# 
# - 標籤編碼。
# - 獨熱編碼。

# ## 標籤編碼（Label encoder）
# 
# 將類別變數的獨一值用 0 到 `n_classes - 1` 的整數表示，可以使用 Scikit-Learn 中的 `LabelEncoder` 轉換。

# In[42]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
pos = player_stats['pos'].values
pos_le = le.fit_transform(pos)
print(pos[:10])
print(pos_le[:10])


# ## 獨熱編碼（One-hot encoder）
# 
# 將類別變數的獨一值用 `(m, n_classes)` 的稀疏矩陣表示，用 `1` 標註是該類，其餘欄位則用 `0` 標註不是該類，可以使用 Scikit-Learn 中的 `OneHotEncoder` 轉換。

# In[43]:


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
pos_ohe = ohe.fit_transform(pos.reshape(-1, 1)).toarray()
print(pos[:10])
print(pos_ohe[:10])


# ## 標籤編碼與獨熱編碼
# 
# - 標籤編碼適合應用於具有量值層級意義、有排列順序的類別變數（例如冷熱可以對應溫度、快慢可以對應速度）與二元分類的情境。
# - 獨熱編碼適合應用於一般無排列順序的類別變數以及多元分類的情境。

# ## 集成模型

# ## 其他常見的分類模型
# 
# - k 最近鄰。
# - 高斯單純貝氏分類器。
# - 決策樹。

# ## 什麼是 k 最近鄰
# 
# - k 最近鄰（k-Nearest Neighbors, KNN）是一種基於資料之間的相似度來決定是否為同一類別的演算方法。
# - 「歐幾里德距離 Euclidean distance」是最常用來量測資料相似度的指標，歐幾里德距離以白話文敘述其實就是直線距離。
# 
# \begin{align}
# d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
# \end{align}

# ## 什麼是 k 最近鄰（續）
# 
# - 更為泛用的距離量測是「明可夫斯基距離 Minkowski distance」。
# - 當式子中的 $p=1$ 時就是曼哈頓距離、$p=2$ 時就是歐幾里德距離。
# 
# \begin{align}
# d(x, y) = \left( \sum_{i=1}^{n} \mid x_i - y_i \mid ^p \right)^{\frac{1}{p}}
# \end{align}

# ## 什麼是 k 最近鄰（續）
# 
# - k 最近鄰會根據預測資料點周遭的 k 個最相似訓練資料點決定分類結果，k 可以由使用者自行決定。
# - 在二元分類的範疇下，k 會選擇一個奇數使得分類結果直接被決定。
# - k 最近鄰模型的訓練與預測在同時間發生，也就是訓練在輸入預測資料時才發生，因此屬於 Lazy learning 的機器學習方法。

# ## 什麼是高斯單純貝氏分類器
# 
# - 高斯單純貝氏分類器是基於貝氏定理的分類模型。
# - 貝氏定理是在先驗機率（Prior probability）的基礎上，納入新事件的資訊來更新先驗機率，得到後驗機率（Posterior probability）的統計分法。
# 
# \begin{align}
# P(y|x_i) = \frac{P(x_i|y) \times P(y)}{P(x_i)} \\
# \text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}
# \end{align}

# ## 什麼是高斯單純貝氏分類器（續）
# 
# - 高斯單純貝氏分類器中的「單純」指的是在計算分類機率時，會假設資料特徵不依賴類別，兩者彼此獨立。
# - 因此實際在計算後驗機率的時候，只需要關注分子的部分。
# 
# \begin{align}
# P(y|x_i) \propto P(x_i|y) P(y) \\
# y^* = argmax_y \, P(x_i|y) P(y) \\
# y^* = argmax_y \, P(y) \prod P(x_i|y)
# \end{align}

# ## 什麼是高斯單純貝氏分類器（續）
# 
# 當特徵為連續型變數時，可以藉由假設變數為常態分配的情況下，以樣本資料的平均數及標準差來計算機率（Likelihood），也就是 $P(x_i|y)$ 能以高斯機率密度函數計算。
# 
# \begin{align}
# P(x_i | y) = \frac{1}{\sqrt{2 \pi \sigma_y^2}} exp \left( -\frac{(x_i - \mu_y)^2}{2 \sigma_y^2} \right)
# \end{align}

# ## 什麼是高斯單純貝氏分類器（續）
# 
# 當特徵數量很多的時候，$\prod P(x_i|y)$ 相乘所算出的機率值會非常小，造成後驗機率趨近於 0，這時可以透過取對數函數來避免。
# 
# \begin{align}
# y^* = argmax_y \, log \left(P(y) \right) + \sum log \left( P(x_i|y) \right)
# \end{align}

# ## 什麼是決策樹
# 
# - 決策樹（Decision tree）是一種利用外型像樹一樣的圖形決策模型，具有快速、可解釋性高的優點。
# - 決策樹需要從資料中尋找合適的「特徵」與「切點」來進行樹的分支，多次分支後企圖讓資料有高差異性的分類。
# - 例如決策是否要跑一場馬拉松：
#     - 氣溫是否低於 15 度？
#         - 否，不要跑。
#         - 是。
#             - 濕度是否低於 60%？
#                 - 否，不要跑。
#                 - 是，要跑。

# ## 什麼是決策樹（續）
# 
# 建立一個決策樹模型必須要考量三個要素：
# 
# 1. 要使用資料中的哪個變數作為特徵。
# 2. 要如何決定特徵的切點。
# 3. 何時要停止分支。

# ## 什麼是決策樹（續）
# 
# - 使用演算法計算資訊增益（Information Gain）、熵（Entropy）、資訊增益率（Information Gain Ratio）或吉尼不純度（Gini Impurity）來決定前述三要素。
# - 因此建構決策樹的演算法可再分為：
#     - ID3(Iterative Dichotomiser 3)
#     - C4.5
#     - C5.0
#     - CART

# ## 使用集成模型讓不同的分類器一起運作
# 
# - 藉由結合多種模型的表現，提升最後的分類結果。
# - 一般有三種常見的架構：
#     - Bagging
#     - Bootstrapping
#     - Stacking

# ## 使用集成模型讓不同的分類器一起運作
# 
# - 藉由結合多種模型的表現，提升最後的分類結果。
# - 常見的結構有：
#     - Bagging
#     - Stacking
#     - Voting
#     - Boosting

# ## 我們以簡單的 Voting 理解集成模型的核心精神

# In[44]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

k_neighbors_clf = KNeighborsClassifier()
k_neighbors_clf.fit(X_train, y_train)
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, y_train)
decision_tree_clf = DecisionTreeClassifier()
decision_tree_clf.fit(X_train, y_train)


# In[45]:


y_hat_knn = k_neighbors_clf.predict(X_valid)
y_hat_gaussian_nb = gaussian_nb.predict(X_valid)
y_hat_decision_tree = decision_tree_clf.predict(X_valid)
print(y_hat_knn)
print(y_hat_gaussian_nb)
print(y_hat_decision_tree)


# In[46]:


y_hat_voted = []
for y_hat_i, y_hat_j, y_hat_k in zip(y_hat_knn, y_hat_gaussian_nb, y_hat_decision_tree):
    if sum((y_hat_i, y_hat_j, y_hat_k)) <= 1:
        y_hat_voted.append(0)
    else:
        y_hat_voted.append(1)
print(np.array(y_hat_voted))


# ## 以隨機森林模型理解 Bagging
# 
# - Bagging 指的是樣本重複抽樣，產生多個子資料集後依序建立多個模型，最後再將所有模型的結果彙整在一起。
# - 隨機森林模型運用 Bagging 與決策樹，多個模型全都都是用決策樹來建模，故得名「森林」。
# - 隨機森林模型在抽樣過程中，不只是對列數（Rows）進行抽樣，同時也會對欄數（Columns）抽樣，因此產生的子集資料，其實是對列跟欄抽樣後的結果。
# - 面對資料中有共線性（Collinearity）跟類別不平衡（Class Imbalance）時採用隨機森林模型，對列抽樣時，可以部份解決類別不平衡來影響預測的問題；對欄抽樣時，可以部份解決共線性來影響預測的問題。

# In[47]:


from sklearn.ensemble import RandomForestClassifier

random_forest_clf = RandomForestClassifier()
random_forest_clf.fit(X_train, y_train)
y_hat_random_forest = random_forest_clf.predict(X_valid)
print(y_hat_random_forest)


# ## 模型選擇

# ## 評估指標的選擇
# 
# - 除了與任務種類相關，也與模型的應用場景有關。
# - 例如即便同屬於疾病的檢測分類模型，針對傳染疾病或罕見疾病所選擇的指標就有可能不同。
# - 這是由於和「誤分類」所衍生出的成本連動所致。

# ## 回歸模型表現的評估指標
# 
# - 均方誤差（Mean squared error）
# - 平均絕對誤差（Mean absolute error）

# ## 使用 Scikit-Learn 定義好的 `mean_squared_error()` 函數可以計算兩個目標向量之間的均方誤差

# In[48]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = player_stats['heightMeters'].values.reshape(-1, 1)
y = player_stats['weightKilograms'].values.astype(float)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
h = LinearRegression()
h.fit(X_train, y_train)
y_hat = h.predict(X_valid)
mse_valid = mean_squared_error(y_valid, y_hat)
mse_valid


# ## 自訂均方誤差函數

# In[49]:


def meanSquaredError(y_true, y_pred):
    error = (y_true - y_pred)
    squared_error = error**2
    mean_squared_error = np.mean(squared_error)
    return mean_squared_error


# In[50]:


mse_valid = meanSquaredError(y_valid, y_hat)
mse_valid


# ## 平均絕對誤差（Mean absolute error）
# 
# - 平均絕對誤差和均方誤差相同之處在於他們都能精確捕捉預測失準的量值。
# - 相異之處在於均方誤差對於預測失準較多的離群值（Outliers）具有放大的效果。
# - 均方誤差適用於離群值會導致錯誤預測的成本更大幅度上升的應用場景。

# ## 使用 Scikit-Learn 定義好的 `mean_absolute_error()` 函數可以計算兩個目標向量之間的平均絕對誤差

# In[51]:


from sklearn.metrics import mean_absolute_error

mae_valid = mean_absolute_error(y_valid, y_hat)
mae_valid


# ## 自訂平均絕對誤差的函數

# In[52]:


def meanAbsoluteError(y_true, y_pred):
    error = (y_true - y_pred)
    absolute_error = np.abs(error)
    mean_absolute_error = np.mean(absolute_error)
    return mean_absolute_error


# In[53]:


mae_valid = meanAbsoluteError(y_valid, y_hat)
mae_valid


# ## 分類器使用的評估指標比迴歸模型為多
# 
# - 準確率（Accuracy）
# - 精確率（Precision）
# - 召回率（Recall）
# - F1-score
# - ...等。

# ## 理解評估分類器指標的設計
# 
# 拆解正確分類 $y^{(valid)} = \hat{y}^{(valid)}$ 與錯誤分類 $y^{(valid)} \neq \hat{y}^{(valid)}$ 的組成。

# ## 正確分類與錯誤分類各自都還能拆解成兩種情境
# 
# - 正確分類
#     - 真陰性（True negative, TN）：$y^{(valid)}=0$ 並且 $\hat{y}^{(valid)}=0$
#     - 真陽性（True positive, TP）：$y^{(valid)}=1$ 並且 $\hat{y}^{(valid)}=1$
# - 錯誤分類
#     - 偽陰性（False negative, FN）：$y^{(valid)}=1$ 並且 $\hat{y}^{(valid)}=0$
#     - 偽陽性（False positive, FP）：$y^{(valid)}=0$ 並且 $\hat{y}^{(valid)}=1$

# ## 混淆矩陣（Confusion matrix）
# 
# 以 $2 \times 2$ 矩陣表達正確、錯誤分類的情境。
# 
# ||$\hat{y}^{(valid)}=0$|$\hat{y}^{(valid)}=1$|
# |---|---|---|
# |$y^{(valid)}=0$|真陰性（True negative, TN）|偽陽性（False positive, FP）|
# |$y^{(valid)}=1$|偽陰性（False negative, FN）|真陽性（True positive, TP）|

# ## 評估指標可以從組成混淆矩陣的四個象限衍生而得
# 
# 使用 Scikit-Learn 定義好的 `confusion_matrix` 函數創造兩個目標向量之間正確分類、錯誤分類所組成的混淆矩陣。

# In[54]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

X = player_stats[['apg', 'rpg']].values
pos = player_stats['pos'].values
y = np.array([0 if p[0] == 'G' else 1 for p in pos])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
h = LogisticRegression()
h.fit(X_train, y_train)
y_hat = h.predict(X_valid)
cm = confusion_matrix(y_valid, y_hat)
cm


# ## 自訂創造混淆矩陣的函數

# In[55]:


def confusionMatrix(y_true, y_pred):
    n_unique = np.unique(y_true).size
    cm = np.zeros((n_unique, n_unique), dtype=int)
    for i in range(n_unique):
        for j in range(n_unique):
            n_obs = np.sum(np.logical_and(y_true == i, y_pred == j))
            cm[i, j] = n_obs
    return cm


# In[56]:


cm = confusionMatrix(y_valid, y_hat)
cm


# ## 準確率（Accuracy）
# 
# 是類別預測任務最常用的評估指標。
# 
# \begin{equation}
# \text{Accuracy} = \frac{\text{TN} + \text{TP}}{\text{TN} + \text{TP} + \text{FN} + \text{FP}}
# \end{equation}

# ## 使用 Scikit-Learn 定義好的 `accuracy_score` 函數可以計算準確率

# In[57]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_valid, y_hat)
accuracy


# ## 準確率不適合評估分類器的場景
# 
# - 陽性事件發生率極低的應用場景，例如罕見疾病或市場黑天鵝事件的預測任務。
# - 如果設計出一個樸素的分類器（Dummy classifier），它以目標向量中出現頻率最高的類別作為預測依據
# - 以 1,000 個觀測值中僅有 1 個陽性的情況舉例，準確率可以達到 0.999，是一個乍看之下非常漂亮的表現。

# In[58]:


y_true = np.zeros(1000, dtype=int)
y_true[-1] = 1
y_hat = np.zeros(1000, dtype=int)
accuracy = accuracy_score(y_true, y_hat)
accuracy


# ## 樸素分類器對預測陽性事件發生完全無用處
# 
# 這時使用精確率（Precision）與召回率（Recall）來進行評估會更加適合。

# ## 精確率
# 
# - 分子是真陽性、分母是真陽性加偽陽性。
# - 它的意涵是分類器在所有預測為陽性的觀測值中，正確預測的觀測值數為多少。
# 
# \begin{align}
# \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
# \end{align}

# ## 召回率
# 
# - 分子是真陽性、分母是真陽性加偽陰性。
# - 它的意涵是分類器在所有陽性的觀測值中，正確預測的觀測值數為多少。
# 
# \begin{align}
# \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
# \end{align}

# ## 精確率與召回率更專注評估分類器對陽性事件的預測能力
# 
# - 兩個指標愈高，代表模型的表現愈好。
# - 精確率如果表現要好除了真陽性高，偽陽性亦要想辦法降低。
# - 召回率如果表現要好除了真陽性高，偽陰性亦要想辦法降低。

# ## 如何選擇採用精確率或召回率
# 
# - 延伸探討偽陽性或偽陰性所衍生的誤分類成本。
# - 採用精確率代表的要盡可能降低偽陽性，這表示的是偽陽性的成本高，意味著是誤判為陽性事件的成本高（例如誤診而進行高風險的手術）。
# - 採用召回率代表的是要儘可能降低偽陰性，這表示的是偽陰性的成本高，意味著是誤判為陰性事件的成本高（例如誤診而導致超級傳播者沒有隔離而進入社區）。

# ## 使用 Scikit-Learn 定義好的 `precision_score()` 與 `recall_score()` 函數可以計算精確率與召回率

# In[59]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# 樸素分類器在精確率和召回率都得到了最低的評估值
precision = precision_score(y_true, y_hat, zero_division=0)
recall = recall_score(y_true, y_hat)
print(precision)
print(recall)


# ## F-score
# 
# 同時將精確率與召回率納入考量，運用一個係數 $\beta$ 加權兩個指標合成為一個指標。
# 
# \begin{equation}
# F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{(\beta^2 \cdot precision) + recall}
# \end{equation}

# ## $\beta$ 係數表示對精確率或召回率的相對重視程度
# 
# - 如果偽陽性的成本遠高於偽陰性的成本，代表百分百重視精確率，這時代入 $\beta = 0$，F-score 就會是精確率。
# - 如果偽陰性的成本遠高於偽陽性的成本，代表百分百重視召回率，這時代入 $\beta = \infty$，F-score 就會是召回率。
# - 如果偽陽性的成本和偽陰性的成本相當，代表兩個指標同等重要，這時代入 $\beta = 1$，F-score 就被稱為 F1-score，指標愈高，代表模型的表現愈好。

# \begin{equation}
# F_{1} = 2 \cdot \frac{precision \cdot recall}{precision + recall}
# \end{equation}

# ## 使用 Scikit-Learn 定義好的 `f1_score()` 函數可以計算 F1-score
# 
# 同樣可以看到樸素分類器依然在 F1-score 獲得了最低的評估值。

# In[60]:


from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_hat)
f1


# ## 自訂計算評估指標的類別 `ClfMetrics`

# ```python
# class ClfMetrics:
#     """
#     This class calculates some of the metrics of classifier including accuracy, precision, recall, f1 according to confusion matrix.
#     Args:
#         y_true (ndarray): 1d-array for true target vector.
#         y_pred (ndarray): 1d-array for predicted target vector.
#     """
#     def __init__(self, y_true, y_pred):
#         self._y_true = y_true
#         self._y_pred = y_pred
# ```

# ```python
#     def confusion_matrix(self):
#         """
#         This function returns the confusion matrix given true/predicted target vectors.
#         """
#         n_unique = np.unique(self._y_true).size
#         cm = np.zeros((n_unique, n_unique), dtype=int)
#         for i in range(n_unique):
#             for j in range(n_unique):
#                 n_obs = np.sum(np.logical_and(self._y_true == i, self._y_pred == j))
#                 cm[i, j] = n_obs
#         self._tn = cm[0, 0]
#         self._tp = cm[1, 1]
#         self._fn = cm[1, 0]
#         self._fp = cm[0, 1]
#         return cm
# ```

# ```python
#     def accuracy_score(self):
#         """
#         This function returns the accuracy score given true/predicted target vectors.
#         """
#         cm = self.confusion_matrix()
#         accuracy = (self._tn + self._tp) / np.sum(cm)
#         return accuracy
#     def precision_score(self):
#         """
#         This function returns the precision score given true/predicted target vectors.
#         """
#         precision = self._tp / (self._tp + self._fp)
#         return precision
# ```

# ```python
#     def recall_score(self):
#         """
#         This function returns the recall score given true/predicted target vectors.
#         """
#         recall = self._tp / (self._tp + self._fn)
#         return recall
#     def f1_score(self, beta=1):
#         """
#         This function returns the f1 score given true/predicted target vectors.
#         Args:
#             beta (int, float): Can be used to generalize from f1 score to f score.
#         """
#         precision = self.precision_score()
#         recall = self.recall_score()
#         f1 = (1 + beta**2)*precision*recall / ((beta**2 * precision) + recall)
#         return f1
# ```

# In[61]:


h = LogisticRegression()
h.fit(X_train, y_train)
y_hat = h.predict(X_valid)


# In[62]:


# 混淆矩陣
clf_metrics = ClfMetrics(y_valid, y_hat)
clf_metrics.confusion_matrix()


# In[63]:


# 準確率
clf_metrics.accuracy_score()


# In[64]:


# 精確率
clf_metrics.precision_score()


# In[65]:


# 召回率
clf_metrics.recall_score()


# In[66]:


# F1-score
clf_metrics.f1_score()


# ## 聚類模型

# ## 什麼是聚類（Cluster）模型
# 
# - 聚類（有時稱為分群）是一種統計方法。這種統計方法常用在商業上、電腦科學、生物學和經濟學等領域。
# - 聚類主要目的是將樣本之間的共同屬性精簡成少數幾個同質性次群體，以相似性形成集群，所謂的相似性通常是以「距離」作為衡量，相對距離愈近，相似程度愈高，分群之後可以使得群內差異小、群間差異大。
# - 常見的聚類方法有 K-means、Hierarchical、DBSCAN 等。

# ## 名詞區別：聚類（分群） vs. 分類
# 
# - 聚類方法屬於非監督式學習（Unsupervised learning），非監督式學習最大的特點就是我們給定的資料中不需要擁有標籤 $y$，換句話說，訓練的資料中沒有給定正確答案，模型會根據我們給定的資料特徵進行分群的操作。
# - 分群是將相同特性之資料歸納為同一群體，以群內差異小，群間差異大作為目標；分類會依據我們事先給定的條件來判定類別，兩者最大的差別在於分群是使用演算法去找出分群的條件，而分類必須事先給定條件設定，依據給定條件去尋找符合之類別。

# ## K 平均數聚類模型
# 
# - 最常見的聚類模型是 K 平均數（K-means），其演算步驟為：
#     - 決定 k 值（分成 k 群）。
#     - 隨機給定 k 個群心（中心點）。
#     - 計算每個樣本與每個群心之距離，並將樣本歸類分配給距離最近的群心。
#     - 通過分配給每個先前群心的所有樣本的平均值來建立新群心。
#     - 重複前述兩個步驟，直到群心不再有太大的變動。

# In[67]:


from sklearn.cluster import KMeans

X = player_stats[['apg', 'rpg']].values
kmeans = KMeans(n_clusters=2, n_init="auto")
kmeans.fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
X_valid = np.array([
    [1, 10],
    [10, 1]
])
y_hat = kmeans.predict(X_valid)
print(y_hat)


# ## 時間序列模型

# ## 什麼是時間序列模型
# 
# - 時間序列是我們周邊時常出現的數據型態，例如：每天的最高溫變化、遊樂園各小時來客數、股價...等。
# - 基本邏輯來說，當數據有隨著時間變化的趨勢，就是時間序列，甚至不一定是跟著時間，基本上只要資料有前後相關性、週期變化現象，就可以用時序的方式處理，適合只有一個維度的簡單型資料。

# ## ARIMA(AutoRegressive Integrated Moving Average)
# 
# - ARIMA 是最常見的時間序列模型：
#     - AutoRegressive 指的是用 $x_{t_{1}}$ 到 $x_{t-1}$ 來預測 $x_t$
#     - Integrated 指的是「差分」，目的是使時間序列資料呈現穩態（Stationery）。
#     - Moving Average 用來描述 AutoRegressive 模型的殘差與 $x_t$ 的關係。

# ## 建立 ARIMA 的標準步驟
# 
# - 透過 Augmented Dickey-Fuller Test 檢定資料是否穩態。
# - 資料若沒有通過穩態檢定，則使用每個時間戳記的資料差 $\delta_t = x_{t} - x_{t-1}$ 再次檢定，直到通過穩態檢定。
# - 記錄前項相減的次數 $d$，作為 ARIMA 模型的參數。
# - 透過 Autocorrelation Function(ACF)決定 $p$ 作為 ARIMA 模型的參數。
# - 透過 Partial Autocorrelation Function(PACF)決定 $q$ 作為 ARIMA 模型的參數。

# ## 取得時間序列資料

# In[68]:


import yfinance as yf

dat = yf.Ticker("VT")
df = dat.history(period="12mo")
df


# In[69]:


ts = df["Close"]
ts


# ## 繪製原始的時間序列資料（未差分）

# In[70]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(ts)
plt.show()


# ## Augmented Dickey-Fuller Test

# In[71]:


from statsmodels.tsa.stattools import adfuller

def adf_test_on_ts(ts) -> None:
    adf_test = adfuller(ts)
    p_value = adf_test[1]
    is_adf_test_passed = p_value < 0.05
    print(f"p-value of Augmented Dickey-Fuller Test: {p_value}")
    print(f"Is Augmented Dickey-Fuller Test passed: {is_adf_test_passed}")

adf_test_on_ts(ts)


# ## 一次差分之後再做 Augmented Dickey-Fuller Test

# In[72]:


close_diff = ts.diff().dropna()
print(close_diff)
adf_test_on_ts(close_diff) # suggests d = 1


# ## 繪製差分一次的時間序列資料

# In[73]:


fig, ax = plt.subplots()
ax.plot(close_diff)
plt.show()


# ## 透過 Autocorrelation Function(ACF)決定 $p$ 作為 ARIMA 模型的參數

# In[74]:


from statsmodels.graphics.tsaplots import plot_acf

plot_acf(close_diff)
plt.show() # suggests p = 1


# ## 透過 Partial Autocorrelation Function(PACF)決定 $q$ 作為 ARIMA 模型的參數

# In[75]:


from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(close_diff) # suggests q = 1
plt.show()


# ## 建立訓練與驗證資料

# In[76]:


train_size = int(len(close_diff) * 0.8)
y_train, y_valid = close_diff.iloc[:train_size], close_diff.iloc[train_size:]


# ## 建立 ARIMA 模型訓練並預測

# In[77]:


from statsmodels.tsa.arima.model import ARIMA

p, d, q = 1, 1, 1
steps_to_be_predicted = y_valid.size
model = ARIMA(y_train.values, order=(p, d, q))
model_fit = model.fit()
forecast = model_fit.get_forecast(steps=steps_to_be_predicted)


# ## 繪圖驗證

# In[78]:


fig, ax = plt.subplots()
ax.plot(y_train.index, y_train.values, label="Train", color="Blue")
ax.plot(y_valid.index, y_valid.values, label="Valid", color="Green")
ax.plot(y_valid.index, forecast.predicted_mean, label="Predicted Mean", color="Orange")
ax.fill_between(y_valid.index, forecast.conf_int()[:, 0], 
                forecast.conf_int()[:, 1], 
                color="Grey", alpha=.25)
plt.legend()
plt.show()


# ## 隨堂練習

# ## 隨堂練習
# 
# <https://colab.research.google.com/github/datainpoint/classroom-fintech-bot-2026/blob/main/06-exercises.ipynb>
