#!/usr/bin/env python
# coding: utf-8

# # 臺灣銀行委辦 AI 人才進階訓練
# 
# > 數據分析，2026-04-10
# 
# [郭耀仁](https://hahow.in/@tonykuoyj?tr=tonykuoyj) | <yaojenkuo@ntu.edu.tw>

# ## 目錄
# 
# - NumPy 入門（P.3）
# - Ndarray 操作（P.32）
# - Pandas 入門（P.44）
# - 資料載入（P.78）
# - DataFrame 操作（P.86）
# - 隨堂練習（P.129）

# ## NumPy 入門

# ## 什麼是 NumPy
# 
# > NumPy 是 Numeric Python 的簡稱，是 Python 最重要的資料科學模組之一。NumPy 創造了 `ndarray` 的資料結構類別以及大量的通用函數與聚合函數，讓 Python 使用者能夠對進行快速的數值計算、使用統計函數、進行線性代數運算以及操作隨機的模擬任務等。
# 
# 來源：<https://numpy.org/>

# ## （沒什麼用的冷知識）NumPy 的前身為兩個模組
# 
# - 1990 年代中期誕生的` Numeric` 與 `Numarray` 模組。
# - NumPy 於 2005 集兩者之大成問世。
# 
# 來源：<https://www.nature.com/articles/s41586-020-2649-2>

# ## 根據說明文件的範例載入
# 
# 來源：<https://numpy.org/doc/stable/user/absolute_beginners.html#how-to-import-numpy>

# In[1]:


import numpy as np


# ## 可以透過兩個屬性檢查版本與安裝路徑
# 
# - `__version__` 屬性檢查版本號。
# - `__file__` 屬性檢查安裝路徑。

# In[2]:


print(np.__version__)
print(np.__file__)


# ## NumPy 的核心功能
# 
# 1. 使用 `ndarray` 來進行數值操作。
# 2. 使用模組定義的函數對 `ndarray` 進行數值運算。
# 3. `ndarray` 是其他資料科學模組 Pandas、Matplotlib 與 Scikit-Learn 的基石。

# ## 從 `list` 創造 `ndarray`

# In[3]:


prime_list = [2, 3, 5, 7, 11]
prime_array = np.array(prime_list)
print(prime_array)
print(type(prime_array))


# ##  利用 NumPy 函數創造內容元素相同的 `ndarray`
# 
# - `np.zeros()`
# - `np.ones()`
# - `np.full()`

# In[4]:


print(np.zeros(5, dtype=int))
print(np.ones(5, dtype=float))
print(np.full(5, 6))


# ## 利用 NumPy 函數創造數列型態的 `ndarray`
# 
# - `np.arange(start, stop, step)`
# - `np.linspace(start, stop, num)` 值得注意的是 `stop` 參數預設為包含。

# In[5]:


print(np.arange(1, 11, 2))
print(np.linspace(1, 9, 5, dtype=int))


# ## 常用的 `ndarray` 屬性
# 
# - `ndarray.ndim` 維度數。
# - `ndarray.shape` 外型。
# - `ndarray.size` 元素個數。
# - `ndarray.dtype` 資料類別。

# In[6]:


prime_list = [2, 3, 5, 7, 11]
prime_array = np.array(prime_list)
print(prime_array.ndim)
print(prime_array.shape)
print(prime_array.size)
print(prime_array.dtype)


# ## 不同維度數的 `ndarray` 有不同的暱稱
# 
# - 零維 `ndarray`：純量（Scalar）。
# - 一維 `ndarray`：向量（Vector）。
# - 二維 `ndarray`：矩陣（Matrix）。
# - 三維或者 n 維 `ndarray`：張量（Tensor）。

# ## 純量、向量、矩陣與張量外型示意圖
# 
# ![Imgur](https://i.imgur.com/81fM6Hf.png)
# 
# 來源：<https://dev.to/juancarlospaco/tensors-for-busy-people-315k>

# In[7]:


scalar = np.array(5566)
print(scalar)
print(scalar.ndim)
print(scalar.shape)


# In[8]:


vector = np.array([5, 5, 6, 6])
print(vector)
print(vector.ndim)
print(vector.shape)


# In[9]:


matrix = np.array([[5, 5],
                   [6, 6],
                   [55, 66]])
print(matrix)
print(matrix.ndim)
print(matrix.shape) # (m, n)


# In[10]:


tensor = np.array([[[5, 5],
                    [6, 6],
                    [55, 66]],
                   [[5, 5],
                    [6, 6],
                    [55, 66]],
                   [[5, 5],
                    [6, 6],
                    [55, 66]],
                   [[5, 5],
                    [6, 6],
                    [55, 66]]])
print(tensor)
print(tensor.ndim)
print(tensor.shape) # (l, m, n)


# ## `ndarray` 與 `list` 相同的地方
# 
# - indexing/slicing 的語法。
# - 能夠以 indexing 更新。

# ## indexing 的語法
# 
# `ndarray` 採用兩個方向的索引機制：
# 
# 1. 由左至右：「從 0 開始」的索引機制。
# 2. 由右至左：「從 -1 開始」的索引機制。

# In[11]:


primes_array = np.array([2, 3, 5, 7, 11])
print("From start to stop:")
print(primes_array[0])
print(primes_array[1])
print(primes_array[2])
print(primes_array[3])
print(primes_array[primes_array.size - 1])
print("From stop to start:")
print(primes_array[-1])
print(primes_array[-2])
print(primes_array[-3])
print(primes_array[-4])
print(primes_array[-primes_array.size])


# ## slicing `[start:stop:step]`
# 
# 除了可以取出特定位置的單個資料值，`ndarray` 也支援擷取特定片段，藉此獲得一個較短長度 `ndarray` 的語法。
# 
# - `start` 起始位置（包含）。
# - `stop` 終止位置（排除）。
# - `step` 間隔。

# In[12]:


print(primes_array[::])   # default
print(primes_array[::2])  # step=2
print(primes_array[:3])   # stop=5, exclusive
print(primes_array[3:])   # start=5, inclusive
print(primes_array[::-1]) # step=-1, reverse


# ## 能夠以 indexing 更新

# In[13]:


primes_array = np.array([2, 3, 5, 7, 11])
print(primes_array)   # before update
primes_array[-1] = 13 # update
print(primes_array)   # after update


# ## `ndarray` 與 `list` 相異的地方
# 
# - indexing 二維以上的 `ndarray` 可以用更便捷的語法 `[i, j, k, ...]`
# - 同質性資料結構類別。
# - 支援元素操作（Elementwise）運算。
# - 支援特殊的 indexing 語法。

# ## indexing 二維以上的 `ndarray` 可以用 `ndarray` 便捷的語法 `[i, j, k, ...]`

# In[14]:


matrix = np.array([[5, 5],
                   [6, 6],
                   [55, 66]])
print(matrix)
print(matrix[2, 1]) # 66 locates at [2, 1]


# In[15]:


print(matrix[:, 1])
print(matrix[:, [1]]) # keep dimension


# ## 同質性資料結構類別

# In[16]:


heterogeneous_list = [False, True, 5566, 55.66, 'Luke Skywalker']
for element in heterogeneous_list:
    print(type(element))


# In[17]:


homogeneous_array = np.array(heterogeneous_list)
for element in homogeneous_array:
    print(type(element))


# ## 支援元素操作（Elementwise）運算

# In[18]:


# list does not support elementwise
primes_list = [2, 3, 5, 7, 11]
try:
    primes_list**2
except TypeError as error_message:
    print(error_message)


# In[19]:


# ndarray supports elementwise
primes_array = np.array(primes_list)
primes_array**2


# ## 支援特殊的 indexing 語法
# 
# - Fancy indexing
# - Boolean indexing

# ## 什麼是 Fancy indexing
# 
# 對應 `ndarray` 時中括號允許傳入 `list`，藉此可以更有彈性地取出 `ndarray` 中的元素。

# In[20]:


primes_list = [2, 3, 5, 7, 11]
try:
    primes_list[[0, 1, 4]]
except TypeError as error_message:
    print(error_message)


# In[21]:


primes_array = np.array([2, 3, 5, 7, 11])
print(primes_array)
print(primes_array[[0, 1, 4]])


# ## 什麼是 Boolean indexing
# 
# 對應 `ndarray` 時中括號允許傳入由 `bool` 組成的相同長度 `list` 或 `ndarray`，藉此可以更有彈性地取出 `ndarray` 中的元素。

# In[22]:


primes_list = [2, 3, 5, 7, 11]
try:
    primes_list[[False, True, True, True, True]]
except TypeError as error_message:
    print(error_message)


# In[23]:


primes_array = np.array([2, 3, 5, 7, 11])
print(primes_array)
print(primes_array[[False, True, True, True, True]])
print(primes_array % 2 == 1)
print(primes_array[primes_array % 2 == 1])


# ## Ndarray 操作

# ## 常用的 `ndarray` 操作
# 
# - 調整外型。
# - 複製。
# - 合併。
# - 分割。

# ## 調整外型
# 
# - `ndarray.reshape()`：調整為指定外型 `(..., m, n)`
# - `ndarray.ravel()`：調整為一維 `(m,)`

# In[24]:


array_range = np.arange(1, 13)
print(array_range)
print(array_range.shape)
print(array_range.reshape(3, 4))
print(array_range.reshape(3, 4).shape)


# ## 在其他維度已經決定時可以方便地指定 `-1` 給最後一個維度

# In[25]:


print(array_range.reshape(3, -1))
print(array_range.reshape(-1, 3))


# ## 使用 `ndarray.ravel()` 調整成一維

# In[26]:


array_range = np.arange(1, 13).reshape(3, -1)
print(array_range.shape)
print(array_range.ndim)
print(array_range.ravel().shape)
print(array_range.ravel().ndim)


# ## 複製
# 
# - 透過物件命名參照並不會真的複製，會讓兩個物件名稱共享一個 `ndarray` 的資料值，但是卻能有不同的外型。
# - 使用 `ndarray.copy()` 明確地複製。

# In[27]:


vector = np.arange(1, 10)
matrix = vector.reshape(3, 3)
matrix[1, 1] = 5566
print(vector)
print(matrix)


# ## 使用 `ndarray.copy()` 明確地複製

# In[28]:


vector = np.arange(1, 10)
matrix = vector.copy().reshape(3, 3)
matrix[1, 1] = 5566
print(vector)
print(matrix)


# ## 合併
# 
# 使用 `np.concatenate()` 函數合併。
# 
# - 指定參數 `axis=0` 垂直合併（預設值）。
# - 指定參數 `axis=1` 水平合併。

# In[29]:


array_a = np.arange(1, 5).reshape(2, 2)
array_b = np.arange(5, 9).reshape(2, 2)
print(np.concatenate((array_a, array_b)))         # default, axis=0
print(np.concatenate((array_a, array_b), axis=1)) # axis=1


# ## 分割
# 
# 使用 `np.split()` 函數分割。
# 
# - 指定參數 `axis=0` 垂直分割（預設值）。
# - 指定參數 `axis=1` 水平分割。

# In[30]:


array_range = np.arange(20).reshape(-1, 2)
upper_array, lower_array = np.split(array_range, 2) # split to 2 ndarrays
print(upper_array)
print(lower_array)


# ## 如果以 `list` 傳入參數則表示分割的索引值

# In[31]:


array_range = np.arange(20).reshape(-1, 2)
upper_array, lower_array = np.split(array_range, [2]) # split on index 2
print(upper_array)
print(lower_array)


# In[32]:


array_range = np.arange(20).reshape(-1, 2)
left_array, right_array = np.split(array_range, 2, axis=1)
print(left_array)
print(right_array)


# ## Pandas 入門

# ## 什麼是 Pandas
# 
# > Pandas 是 Python 處理表格式資料（Tabular data）的第三方模組，它創造了 `Index`、`Series` 與 `DataFrame` 的資料結構類別，讓 Python 在面對表格式資料時能夠用更直覺的觀念操作。
# 
# 來源：<https://github.com/pandas-dev/pandas>

# ## （沒什麼用的冷知識）Pandas 跟熊貓「沒有關係」
# 
# 1. **Pan**el(自從版本 0.20.0 之後棄用)
# 2. **Da**taFrame
# 3. **S**eries
# 
# ![](https://media.giphy.com/media/46Zj6ze2Z2t4k/giphy.gif)
# 
# 來源：<https://media.giphy.com/media/46Zj6ze2Z2t4k/giphy.gif>

# ## 根據說明文件的範例載入
# 
# 來源：<https://pandas.pydata.org/docs/user_guide/10min.html>

# In[33]:


import pandas as pd


# ## 可以透過兩個屬性檢查版本號與安裝路徑
# 
# - `__version__` 屬性檢查版本號。
# - `__file__` 屬性檢查安裝路徑。

# In[34]:


print(pd.__version__)
print(pd.__file__)


# ## 入門 Pandas 的第一步就是掌握 `Index`、`ndarray`、`Series` 與 `DataFrame` 四個資料結構類別彼此之間的關係
# 
# - `Series` 由 `Index` 與 `ndarray` 組合而成。
# - `DataFrame` 由數個共享同一個 `Index` 的 `Series` 組合而成。

# ## Pandas 的 `Index` 類別
# 
# 使用 `pd.Index()` 函數創造 `Index` 類別的實例。

# In[35]:


import numpy as np


# In[36]:


primes_array = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
prime_indexes = pd.Index(primes_array)
print(prime_indexes)
print(type(prime_indexes))


# ## `Index` 的基礎屬性
# 
# - `Index.dtype` 資料類別。
# - `Index.size` 元素個數。

# In[37]:


print(prime_indexes.dtype)
print(prime_indexes.size)


# ## `Index` 類別結合 Python 內建的 `tuple` 與 `set` 兩種資料結構類別的特性
# 
# - 具有 `tuple` 無法更動的特性。
# - 具有 `set` 集合運算的特性。 

# ## `Index` 類別具有 `tuple` 無法更動的特性

# In[38]:


# Index has the characteristics of a tuple
primes_array = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
prime_indexes = pd.Index(primes_array)
try:
    prime_indexes[-1] = 31
except TypeError as error_message:
    print(error_message)


# ## `Index` 類別具有 `set` 集合運算的特性

# In[39]:


# Index has the characteristics of a set
primes_array = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
prime_indexes = pd.Index(primes_array)
odd_indexes = pd.Index(np.arange(1, 30, 2))
print(prime_indexes)
print(odd_indexes)


# ## `Index` 具有與 `set` 同樣名稱的集合運算方法
# 
# - `Index.intersection()` 交集。
# - `Index.union()` 聯集。
# - `Index.difference()` 差集。
# - `Index.symmetric_difference()` 對稱差集。

# In[40]:


# Set operations of Index
print(prime_indexes.intersection(odd_indexes))
print(prime_indexes.union(odd_indexes))
print(prime_indexes.difference(odd_indexes))
print(odd_indexes.difference(prime_indexes))
print(prime_indexes.symmetric_difference(odd_indexes))


# ## Pandas 的 `Series` 類別
# 
# 使用 `pd.Series()` 函數創造 `Series` 類別的實例。

# In[41]:


months_array = np.arange(1, 13)
months_series = pd.Series(months_array)
print(months_series)
print(type(months_series))


# ## `Series` 的基礎屬性與方法
# 
# - `Series.dtype` 資料類別。
# - `Series.size` 元素個數。
# - `Series.index` 取出 `Series` 的 `Index` 部分。  
# - `Series.values` 取出 `Series` 的 `ndarray` 部分。
# - `Series.astype()` 轉換 `Series` 的資料類別。

# In[42]:


print(months_series.dtype)
print(months_series.size)


# ## `Series` 由 `Index` 與 `ndarray` 組合而成

# In[43]:


print(months_series.index)
print(type(months_series.index))


# In[44]:


print(months_series.values)
print(type(months_series.values))


# ## 調整 `Series` 的 `Index`
# 
# - 在建立的時候指定。
# - 透過 `Series.index` 更新。

# ## 在建立的時候指定

# In[45]:


months_abbreviation = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
months_series = pd.Series(months_array, index=months_abbreviation)
months_series


# ## 透過 `Series.index` 更新

# In[46]:


months_abbreviation = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
months_series = pd.Series(months_array)
months_series.index = months_abbreviation
months_series


# ## 如何取出 `Series` 中的元素
# 
# - 以元素位置 indexing/slicing
# - 以 `Index` indexing/slicing

# ## 以元素位置 indexing/slicing

# In[47]:


print(months_series[0])
print(months_series[:3])


# ## 以 `Index` indexing/slicing

# In[48]:


print(months_series["JAN"])
print(months_series["JAN":"MAR"])


# ## 轉換 `Series` 的資料類別
# 
# - 在建立的時候指定。
# - 透過 `Series.astype()` 轉換 `Series` 的資料類別。

# ## 在建立的時候指定

# In[49]:


months_series = pd.Series(months_array, dtype=float)
months_series


# ## `Series.astype()` 轉換 `Series` 的資料類別

# In[50]:


months_series = pd.Series(months_array)
print(months_series.dtype)
print(months_series.astype(float))


# ## Pandas 的 `DataFrame` 類別
# 
# 使用 `pd.DataFrame()` 函數創造 `DataFrame` 類別的實例。
# 
# - 輸入以欄為基準（Column-based）的資料內容。
# - 輸入以列為基準（Row-based）的資料內容。

# ## 輸入以欄為基準（Column-based）的資料內容

# In[51]:


movie_df = pd.DataFrame()
movie_df["title"] = ["The Shawshank Redemption", "The Dark Knight", "Schindler's List", "Forrest Gump", "Inception"]
movie_df["imdb_rating"] = [9.3, 9.0, 8.9, 8.8, 8.7]
movie_df["release_year"] = [1994, 2008, 1993, 1994, 2010]
print(movie_df)
print(type(movie_df))


# ## 輸入以列為基準（Row-based）的資料內容

# In[52]:


movies = [
    {"title": "The Shawshank Redemption", "imdb_rating": 9.3, "release_year": 1994},
    {"title": "The Dark Knight", "imdb_rating": 9.0, "release_year": 2008},
    {"title": "Schindler's List", "imdb_rating": 8.9, "release_year": 1993},
    {"title": "Forrest Gump", "imdb_rating": 8.8, "release_year": 1994},
    {"title": "Inception", "imdb_rating": 8.7, "release_year": 2010},
]
movie_df = pd.DataFrame(movies)
print(movie_df)
print(type(movie_df))


# ## Jupyter Notebook 針對 `DataFrame` 類別有特別的顯示外觀

# In[53]:


movie_df


# ## `DataFrame` 的基礎屬性
# 
# - `DataFrame.dtypes` 資料類別。
# - `DataFrame.shape` 外型。
# - `DataFrame.index` 取出列標籤（row labels）部分。  
# - `DataFrame.columns` 取出欄標籤（column labels）的部分。

# In[54]:


print(movie_df.dtypes)
print(movie_df.shape)
print(movie_df.index)
print(movie_df.columns)


# ## `DataFrame` 由數個 `Series` 共享同一個 `Index` 組成

# In[55]:


print(type(movie_df.index))
print(type(movie_df["title"]))
print(type(movie_df["imdb_rating"]))
print(type(movie_df["release_year"]))


# ## `DataFrame` 的基礎方法
# 
# - `DataFrame.head(n)` 檢視前 `n` 列。
# - `DataFrame.tail(n)` 檢視後 `n` 列。
# - `DataFrame.describe()`  檢視數值欄位的描述性統計。
# - `DataFrame.info()` 檢視詳細資訊。

# ## 檢視前 `n` 列、後 `n` 列
# 
# - `DataFrame.head(n)` 檢視前 `n` 列。
# - `DataFrame.tail(n)` 檢視後 `n` 列。

# In[56]:


movie_df.head(3)


# In[57]:


movie_df.tail(2)


# ## `DataFrame.describe()` 檢視數值欄位的描述性統計

# In[58]:


movie_df.describe()


# ## `DataFrame.info()` 檢視詳細資訊

# In[59]:


movie_df.info()


# ## 資料載入

# ## 常見的來源資料格式
# 
# 1. 純文字檔案。
# 2. 試算表。
# 3. 關聯式資料庫中的資料表。

# ## 什麼是純文字檔案
# 
# 只有文字所構成的電腦檔案，不包含字型的樣式或者段落標記，能夠使用最簡單的文字編輯器（例如 Windows 的「記事本」、macOS 的 TextEdit）直接開啟檢視。

# ## 更適合程式設計與資料科學的文字編輯器
# 
# - [Visual Studio Code](https://code.visualstudio.com/)
# - [Atom](https://atom.io/)
# - [Sublime Text](https://www.sublimetext.com/)
# - [Notepad ++(for Windows only)](https://notepad-plus-plus.org/) 

# ## 不同格式的純文字檔案
# 
# - 非結構化的純文字檔案。
# - JSON(JavaScript Object Notation)
# - 特定符號分隔的純文字檔案。

# ## 特定符號分隔的純文字檔案
# 
# - 透過特定符號分隔欄位的結構化資料。
# - 常見的有逗號（`,`）、分號（`;`）、Tab 鍵（`\t`）等。
# - 最廣泛使用的是逗號，因此有特定的副檔名 `.csv` 意指逗號分隔值（Comma-separated values）。

# ## 使用 `pd.read_csv()` 函數載入

# In[60]:


get_ipython().system('wget -Nq https://raw.githubusercontent.com/datainpoint/classroom-fintech-bot-2026/refs/heads/main/movies.csv')


# In[61]:


movies = pd.read_csv("movies.csv")
movies.head() # just show the first 5 rows


# ## 其他特定符號分隔的純文字檔依然可以使用 `pd.read_csv()` 函數
# 
# 指定參數 `sep=";"|"\t"`
# 
# 來源：<https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>

# ## DataFrame 操作

# ## `DataFrame` 是有兩個維度的資料結構
# 
# - 第一個維度稱為觀測值（Observations），有時亦稱為列（Rows）
# - 第二個維度稱為變數（Variables），有時亦稱為欄（Columns）
# - 我們習慣以 `(m, n)` 或者 `m x n` 來描述一個具有 `m` 列觀測值、`n` 欄變數的 `DataFrame`

# In[62]:


print(movies.shape)
print(movies.head().shape)
movies.head() # just show the first 5 rows


# ## `DataFrame` 與二維 `ndarray` 不同的地方
# 
# - `DataFrame` 的每個變數可以是異質的。
# - `DataFrame` 的觀測值具有列標籤（row-label）、變數具有欄標籤（column-label）

# In[63]:


print(movies.dtypes)  # heterogeneous 
print(movies.index)   # row-label
print(movies.columns) # column-label


# ## Pandas 使用更直觀的概念操作資料
# 
# - 如何定義「更直觀」？
#     - 像操作試算表一般（Spreadsheet-like）
#     - 像使用結構化查詢語言一般（SQL-like）

# ## 以 `DataFrame["column"]` 選擇欄位成為外型 `(m,)` 的 `Series`

# In[64]:


print(type(movies["title"]))
print(movies["title"].shape)
movies["title"]


# ## 以 `DataFrame[["column"]]` 選擇欄位成為外型 `(m, 1)` 的 `DataFrame`

# In[65]:


print(type(movies[["title"]]))
print(movies[["title"]].shape)
movies[["title"]]


# ## 以 `DataFrame[["column_0", "column_1", ...]]` 選擇多個欄位成為外型 `(m, n)` 的 `DataFrame`
# 
# 運用 Fancy indexing 於欄位的選擇。

# In[66]:


movies[["title", "director", "release_year", "rating"]]


# ## 以 `DataFrame.loc[:, [column_0, column_1, ...]]` 選擇多個欄位成為外型 `(m, n)` 的 `DataFrame`
# 
# - `loc` 是透過資料位置（Location）指定，也就是根據欄標籤選擇。
# - `:` 表示不針對資料列篩選。

# In[67]:


movies.loc[:, ["title", "director", "release_year", "rating"]]


# ## 以 `DataFrame.iloc[:, [0, 1, ...]]` 選擇多個欄位成為外型 `(m, n)` 的 `DataFrame`
# 
# - `iloc` 是透過資料整數位置（Integer location）指定，也就是根據欄的整數位置選擇。
# - `:` 表示不針對資料列篩選。

# In[68]:


movies.iloc[:, [1, 4, 2, 3]]


# ## 善用 `Series` 的特性
# 
# - `Series` 是由 `Index` 與 `ndarray` 組合而成，具備了 `ndarray` 的特性。
# - 善用元素操作（Elementwise）運算。
# - 善用特殊的 indexing 語法：Fancy indexing/Boolean indexing

# ## 透過列標籤（row-label）篩選資料列
# 
# - 在 `movies` 中魔戒三部曲分別位於列標籤 6, 9, 13
#     - `loc` 是透過資料位置（Location）指定，也就是根據列標籤篩選。
#     - `:` 表示不針對欄位選擇。
# - 運用 Fancy indexing 於資料列的篩選。

# In[69]:


movies.loc[[6, 9, 13], :]


# ## 以 `DataFrame.iloc[[0, 1, ...], :]` 選擇多個資料列
# 
# - 在 `lord_of_the_rings` 中第一集與第三集分別位於第 0, 1 列。
#     - `iloc` 是透過資料整數位置（Integer location）指定，也就是根據列的整數位置篩選。
#     - `:` 表示不針對欄位選擇。
# - 運用 Fancy indexing 於資料列的篩選。

# In[70]:


lord_of_the_rings = movies.loc[[6, 9, 13], :]
lord_of_the_rings.iloc[[0, 1], :]


# ## 區分 `DataFrame` 的兩種索引語法
# 
# - `DataFrame.loc[row-label, column-label]` 以列、欄標籤為準。
# - `DataFrame.iloc[row-integer-location, column-integer-location]` 以列、欄整數位置為準。

# ## 透過條件敘述以 `DataFrame[booleans]` 篩選資料列
# 
# - 運用 Boolean indexing 於資料列的篩選。
# - 熟悉之後會直接將條件敘述寫在中括號裡。

# In[71]:


boolean_series = movies["director"] == "Peter Jackson"
movies[boolean_series] # movies[movies["director"] == "Peter Jackson"]


# ## 如何負面表列（negate）由 `bool` 組成的 Series
# 
# - 使用相反的關係運算符 `==` vs. `!=`
# - 使用 `~` 運算符。

# In[72]:


boolean_series = movies["director"] != "Peter Jackson"
print(boolean_series.sum())
boolean_series = movies["director"] == "Peter Jackson"
print((~boolean_series).sum())


# ## 加入多個條件敘述篩選資料列
# 
# - 運用 `&` 運算符**交集**多個條件敘述。
# - 運用 `|` 運算符**聯集**多個條件敘述。

# ## 運用 `&` 運算符交集多個條件敘述

# In[73]:


(movies["release_year"] == 1994) & (movies["rating"] >= 8.8)


# In[74]:


movies[(movies["release_year"] == 1994) & (movies["rating"] >= 8.8)] # movies released in 1994 with amazing rating score


# ## 運用 `|` 運算符聯集多個條件敘述

# In[75]:


(movies["release_year"] == 1994) | (movies["rating"] >= 8.8)


# In[76]:


movies[(movies["release_year"] == 1994) | (movies["rating"] >= 8.8)]


# ## 運用 `Series.isin()` 聯集單一變數的多個條件

# In[77]:


movies[movies["director"].isin(["Peter Jackson", "Quentin Tarantino"])]


# In[78]:


movies[movies["release_year"].isin([1994, 2008])]


# ## 如何判斷條件敘述的交集或聯集
# 
# - 運用語言邏輯思考條件的結合為「和」還是「或」，「和」為交集、「或」為聯集。
# - 運用資料列數思考條件的結合要「縮減」還是「擴增」，「縮減」為交集、「擴增」為聯集。

# ## 兩個排序方式
# 
# 1. 遞增（又稱升冪）排序，預設的排序方式。
# 2. 遞減（又稱降冪）排序。

# ## 使用 `DataFrame` 的兩個方法排序
# 
# - `DataFrame.sort_index()` 依列標籤排序。
# - `DataFrame.sort_values()` 依欄位排序。

# ## `DataFrame.sort_index()` 依列標籤排序 
# 
# 預設 `ascending=True`

# In[79]:


movies.sort_index()


# In[80]:


movies.sort_index(ascending=False)


# ## `DataFrame.sort_values()` 依欄位排序
# 
# - 預設 `ascending=True`
# - 數值由小到大、英文由 A 到 Z

# In[81]:


movies.sort_values("release_year")


# In[82]:


movies.sort_values("release_year", ascending=False)


# ## 傳入 `list` 指定多個欄位與排序方式

# In[83]:


movies.sort_values(["release_year", "title"], ascending=[False, True])


# ## 使用 `Series` 的聚合方法取得欄位摘要
# 
# - `Series.count()` 不含未定義值的列數
# - `Series.sum()` 加總
# - `Series.max()` 最大值
# - `Series.min()` 最小值
# - `Series.mean()` 平均
# - ...等。

# ## 片長 `runtime` 的摘要

# In[84]:


print(movies["runtime"].max())
print(movies["runtime"].min())


# ## IMDb 評等 `rating` 的摘要

# In[85]:


print(movies["rating"].max())
print(movies["rating"].min())
print(movies["rating"].mean())


# ## 使用 `DataFrame.groupby()` 獲得排序後的獨一值 `DataFrameGroupBy` 類別

# In[86]:


print(movies.groupby("director"))


# ## 分組後接續選擇欄位以及聚合方法

# In[87]:


movies.groupby("director")["title"].count() # number of movies by director


# In[88]:


movies.groupby("director")["rating"].mean() # average rating by director


# ## 善用三個技巧衍生計算欄位
# 
# 1. 元素操作（Elementwise）運算。
# 2. 使用函數或 `Series` 的方法。
# 3. 使用 `Series.map()`

# ## 元素操作（Elementwise）運算

# In[89]:


print(movies["runtime"] // 60) # hours
print(movies["runtime"] % 60)  # minutes


# ## 使用函數或 `Series` 的方法

# In[90]:


hours = (movies["runtime"] // 60).astype(str)
minutes = (movies["runtime"] % 60).astype(str)
hours.str.cat(minutes, sep=":") # hours:minutes


# ## 使用 `Series.map()`

# In[91]:


def mins_to_hourmins(x: int) -> str:
    hours = str(x // 60)
    minutes = str(x % 60)
    return f"{hours.zfill(2)}:{minutes.zfill(2)}" # 2 digits zero-filled

runtime_hours_mins = movies["runtime"].map(mins_to_hourmins)
runtime_hours_mins


# ## 使用 `DataFrame.insert()` 新增變數
# 
# 留意 `DataFrame.insert()` 更新的機制是更新物件本身並回傳 `None`

# In[92]:


print(movies.columns)
n = movies.shape[1]
movies.insert(n, "runtime_hours_mins", runtime_hours_mins)
print(movies.columns)
movies.head()


# ## 如何調整列標籤與欄標籤
# 
# - 使用 `DataFrame.set_index()` 指定欄位取代目前的列標籤。
# - 使用 `DataFrame.reset_index()` 重設列標籤。
# - 指定 `DataFrame.columns` 調整欄標籤。

# ## 使用 `DataFrame.set_index()` 指定欄位取代目前的列標籤

# In[93]:


movies.set_index("title")


# ## 使用 `DataFrame.reset_index()` 重設列標籤
# 
# 預設以 `RangeIndex` 重設後將原本的列標籤變為第零個欄位。

# In[94]:


movies.set_index("title").reset_index()


# ## 使用 `DataFrame.reset_index()` 重設列標籤（續）
# 
# 設定參數 `drop=True` 以 `RangeIndex` 重設後捨棄原本的列標籤。

# In[95]:


movies.set_index("title").reset_index(drop=True)


# ## 指定 `DataFrame.columns` 調整欄標籤

# In[96]:


print(movies.columns)
movies.columns = [column.upper() for column in movies.columns]
print(movies.columns)


# ## 隨堂練習

# ## 隨堂練習
# 
# <https://colab.research.google.com/github/datainpoint/classroom-fintech-bot-2026/blob/main/04-exercises.ipynb>
