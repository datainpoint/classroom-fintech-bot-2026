#!/usr/bin/env python
# coding: utf-8

# # 臺灣銀行委辦 AI 人才進階訓練
# 
# > Python 程式設計（一），2026-03-19
# 
# [郭耀仁](https://hahow.in/@tonykuoyj?tr=tonykuoyj) | <yaojenkuo@ntu.edu.tw>

# ## 目錄
# 
# - 關於 Python（P.3）
# - 建立開發環境（P.17）
# - 哈囉世界（P.23）
# - Python 禪學（P.36）
# - 賦值與註解（P.39）
# - 隨堂練習（P.48）

# ## 關於 Python

# ## Less is more
# 
# - 雖然我們可能從很多資訊來源聽到 Python 簡單易學，但是對於初學者而言，Python 程式設計與資料科學是有難度的。
# - Python 作為一個泛用（General-purposed）程式語言，功能涵蓋面向非常完整。
# - 課程設計要避免貪多嚼不爛、蜻蜓點水的錯誤。
# - 「選擇什麼不學」有時候比「選擇要學什麼」更重要：Learn enough Python is enough(for a newbie.)

# ## 一言以蔽之
# 
# > The way Python works is pretty straightforward, you apply functions to objects or you call methods of objects.
# >
# > Yao-Jen Kuo's revision on Greg Martin

# ## 初學 Python 就上手
# 
# - 選擇一個有興趣的應用情境出發。
# - 掌握基礎語法。
# - 掌握不同資料型別的特性。
# - 掌握不同資料結構的特性。
# - 掌握流程控制。
# - 知道如何組織程式碼。

# ## 掌握基礎語法
# 
# - 重要的 Python 參考文件：
#     - 內建函數一覽：<https://docs.python.org/3/library/functions.html>
#     - 標準模組一覽：<https://docs.python.org/3/library/index.html>
#     - 寫作風格：<https://peps.python.org/pep-0008>
#     - 保留字：應用 `help()` 函數於 `"keywords"` 字串。

# ## 掌握基礎語法（續）
# 
# - 如何載入模組並且給予別名。
# - 如何賦値。
# - 如何載入資料。
# - 如何善用 `print()`、`type()`、`len()` 與 `shape` 檢視物件。

# ## 掌握不同資料型別的特性
# 
# - `str`
# - `int`
# - `float`
# - `bool`
# - `None`

# ## 掌握不同資料結構的特性
# 
# - 內建資料結構
#     - `list`
#     - `tuple`
#     - `dict`
#     - `set`
# - 資料科學資料結構
#     - `ndarray`
#     - `Series`
#     - `DataFrame`

# ## 掌握流程控制
# 
# - 程式分支
#     - `if`
#     - `if...else...`
#     - `if...elif...else...`
# - 迴圈
#     - `for`
#     - `while`

# ## 知道如何組織程式碼
# 
# - 熟練地使用函數/類別/模組。
# - 能夠自行定義函數/類別/模組將可能需要反覆使用的程式碼組織起來。
# - 完成本次訓練課程的同學應該要能達成「熟練地使用函數/類別/模組」，並準備好在不久的將來著手學習自行定義函數/類別/模組。

# ## 現代資料科學：以程式設計做資料科學的應用
# 
# ![](r-for-data-science.png)
# 
# 來源：[R for Data Science](https://r4ds.had.co.nz)

# ## 有很多程式語言可以做到前述六項的資料科學應用，為什麼選 Python
# 
# - R
# - Matlab
# - Julia
# - SAS
# - Scala
# - ...etc.

# ## 選擇最多人使用、推薦與使用人數上升的程式語言
# 
# - [2022 Kaggle ML & DS Survey](https://www.kaggle.com/c/kaggle-survey-2022)
# - [Stack Overflow Trends](https://insights.stackoverflow.com/trends?tags=python%2Cr%2Cmatlab%2Cjulia%2Csas%2Cscala)
# - [TIOBE Index](https://www.tiobe.com/tiobe-index/)

# ## 關於 Python 的二三事
# 
# 1. Python 的作者是荷蘭電腦科學家 Guido van Rossum
# 2. Python 的命名源於 Guido van Rossum 非常喜歡電視喜劇 Monty Python's Flying Circus
# 3. Python 的第一版釋出於 1991 年。

# ## 建立開發環境

# ## 寫作與執行 Python 程式需要三種類型的軟體
# 
# 1. 純文字編輯器：寫作程式的軟體，例如記事本、Visual Studio Code 或 Notepad++。
# 2. 終端機：執行程式的軟體，例如 Windows 的命令提示字元、macOS 的 Terminal。
# 3. Python 直譯器：將 Python 程式翻譯為電腦語言的軟體。

# ## 將這三種類型的軟體整合在一起的軟體叫做整合開發環境（Integrated Development Environment, IDE）
# 
# 受歡迎的 Python 整合開發環境有：
# 
# - JupyterLab/Jupyter Notebook
# - Visual Studio Code
# - PyCharm
# - Spyder
# - ...etc.

# ## 安裝 Miniconda 與 Visual Studio Code 建立開發環境
# 
# - [Miniconda](https://www.anaconda.com/download)
# - [Visual Studio Code](https://code.visualstudio.com/)

# ## 課程中我們將在 Google Colab 寫作與執行 Python 程式
# 
# 透過即時互動的筆記本介面寫作與執行：<https://colab.research.google.com>

# ## 初登場的兩個 Python 程式
# 
# 1. 哈囉世界。
# 2. Python 禪學（The Zen of Python）。

# ## 哈囉世界

# ## 哈囉世界
# 
# - `print()` 是 Python 的內建函數，可以將小括號中的輸入印出。
# - `"Hello world!"` 是屬於 `str` 類別的字面值（Literal value）。

# In[1]:


print("Hello world!")


# ## 哈囉世界
# 
# - 哈囉世界中的 `print()` 是什麼？
# - 哈囉世界中的 `"Hello world!"` 是什麼？

# In[2]:


print("Hello world!")


# ## 哈囉世界中的 `print()` 是什麼
# 
# - `print()` 是 Python 的內建函數。
# - 內建函數是不需要先行「定義」就可以使用的函數。

# ## 什麼是函數
# 
# 一段被賦予名稱的程式碼，能夠完成某一個文字處理或者數值計算任務，在使用函數之前，必須先確定這個函數在執行的環境中已經被定義妥善。

# ## 函數的組成有五個要件
# 
# 1. 函數名稱。
# 2. 輸入
# 3. 參數。
# 4. 運算處理邏輯。
# 5. 輸出。

# ## 函數有四個來源
# 
# 1. 來自內建函數。
# 2. 來自標準模組。
# 3. 來自第三方模組。
# 4. 來自使用者的定義。

# ## 來自內建函數可以直接使用
# 
# - 哪些內建函數可以直接使用：<https://docs.python.org/3/library/functions.html>
# - 使用方式為輸入資料或引數（Parameters）到函數名稱後的小括號。

# In[3]:


abs(-55) # -55 as data


# ## 資料與引數的輸入方式
# 
# 1. 依照位置輸入（Positional arguments）。
# 2. 依照參數名稱輸入（Keyword arguments）。

# ## 依照位置輸入（Positional arguments）

# In[4]:


pow(5, 2) # 5 as base, 2 as exp


# In[5]:


pow(2, 5) # 2 as base, 5 as exp


# ## 依照參數名稱輸入（Keyword arguments）

# In[6]:


pow(exp=2, base=5) # 5 as base, 2 as exp


# ## 使用者定義的函數
# 
# ```python
# def function_name(INPUTS: TYPE, ARGUMENTS: TYPE) -> TYPE:
#     ### BEGIN SOLUTION
#     OUTPUTS = INPUTS (+-*/...) ARGUMENTS
#     return OUTPUTS
#     ### END SOLUTION
# ```

# ## 現階段我們先瞭解一些關於自行定義函數的組成
# 
# - `def` 保留字用來定義函數的名稱。
# - 縮排部分稱為程式區塊（Code block），是函數的主體，也是練習題要學員運用預期輸入與參數來完成的部分。
# - 不要忘記把函數的預期輸出寫在 `return` 保留字後。
# - 函數的類別提示（Typing）並不是必要的，但它能幫助我們更快理解練習題。

# ## Python 禪學

# ## Python 禪學（The Zen of Python）
# 
# - `import` 是 Python 的保留字（Keywords），可以載入模組。
# - `this` 是 Python 的一個標準模組，可以印出 Python 禪學。

# In[7]:


import this


# ## （沒什麼用的冷知識）Python 禪學在 `this` 模組中的內容是經過 ROT13 加密的文字
# 
# ```
# Gur Mra bs Clguba, ol Gvz Crgref
# 
# Ornhgvshy vf orggre guna htyl.
# Rkcyvpvg vf orggre guna vzcyvpvg.
# Fvzcyr vf orggre guna pbzcyrk.
# Pbzcyrk vf orggre guna pbzcyvpngrq.
# Syng vf orggre guna arfgrq.
# Fcnefr vf orggre guna qrafr.
# Ernqnovyvgl pbhagf.
# Fcrpvny pnfrf nera'g fcrpvny rabhtu gb oernx gur ehyrf.
# Nygubhtu cenpgvpnyvgl orngf chevgl.
# Reebef fubhyq arire cnff fvyragyl.
# Hayrff rkcyvpvgyl fvyraprq.
# Va gur snpr bs nzovthvgl, ershfr gur grzcgngvba gb thrff.
# Gurer fubhyq or bar-- naq cersrenoyl bayl bar --boivbhf jnl gb qb vg.
# Nygubhtu gung jnl znl abg or boivbhf ng svefg hayrff lbh'er Qhgpu.
# Abj vf orggre guna arire.
# Nygubhtu arire vf bsgra orggre guna *evtug* abj.
# Vs gur vzcyrzragngvba vf uneq gb rkcynva, vg'f n onq vqrn.
# Vs gur vzcyrzragngvba vf rnfl gb rkcynva, vg znl or n tbbq vqrn.
# Anzrfcnprf ner bar ubaxvat terng vqrn -- yrg'f qb zber bs gubfr!
# ```

# ## 賦值與註解

# ## 哈囉世界中的 `"Hello world!"` 是什麼
# 
# - `"Hello world!"` 是 `str` 類別的字面值（Literal value）。
# - 除了 `str` 類別，字面值也可以是其他的資料或資料結構類別。
# - 單純的字面值不太實用，更好的做法是以一個物件名稱去參照字面值！

# In[8]:


hello_world = "Hello world!"
hello_world


# ## 如何說明 `hello_world = "Hello world!"`
# 
# - `hello_world` 物件是 `str` 類別的實例（Instance）。
# - 我們可以使用 `=` 符號讓 `object_name` 成為 `literal_value` 類別的實例，供後續的程式使用。
# - `hello_world = "Hello world!"` 就是所謂的「賦值」。
# 
# ```python
# # 賦值
# object_name = literal_value
# ```

# ## 筆記本的顯示規則
# 
# - 會將程式儲存格最後一列的字面值、物件或函數輸出。
# - 如果想要有多個輸出，必須明確地使用 `print()` 函數。

# ## 常見的筆記本使用習慣
# 
# - 如果程式儲存格只需要一個輸出，直接參照字面值、物件或函數。
# - 如果程式儲存格需要多個輸出，全部都使用 `print()` 函數。

# In[9]:


# Single output in a code cell
hello_world = "Hello, world!"
hello_world


# In[10]:


# Multiple outputs in a code cell
print(hello_world)
print(hello_world)
print(hello_world)


# ## 物件與函數的命名規則
# 
# - 使用全小寫英文，採用蛇形命名法（Snake case），不同單字之間以底線 `_` 相隔。
# - 不能使用保留字作命名。
# - 使用單數名詞為資料類別的物件命名、使用複數名詞為資料結構類別的物件命名、使用動詞為函數或方法命名，盡量讓名稱簡潔且具有意義。
# - 不要使用內建函數作物件的命名，避免覆蓋內建函數的功能。
# 
# 來源：[PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/#function-and-variable-names)

# ## 什麼是保留字
# 
# - 保留字是具有特殊作用的指令。
# - 目前有看過的 `import`、`def` 與 `return` 等都是 Python 的保留字。
# - Python 的保留字一覽：<https://docs.python.org/3/reference/lexical_analysis.html#keywords>
# 
# ```python
# help("keywords")
# ```

# ## 有時候我們在程式碼之中會看到用來解釋的說明文字
# 
# - 註解（Comments）是口語化的文字敘述，以 `#` 標記，並不能夠被翻譯成電腦語言。
# - 註解可以細分為單行註解、行末註解。

# In[12]:


# A hello world example
hello_world = "Hello world!" # hello_world is an instance of str class
hello_world                  # show hello_world object


# ## 隨堂練習

# ## 隨堂練習
# 
# <https://colab.research.google.com/github/datainpoint/classroom-fintech-bot-2026/blob/main/01-exercises.ipynb>
