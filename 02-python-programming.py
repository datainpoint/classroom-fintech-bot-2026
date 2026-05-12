#!/usr/bin/env python
# coding: utf-8

# # 臺灣銀行委辦 AI 人才進階訓練
# 
# > Python 程式設計（二），2026-03-26
# 
# [郭耀仁](https://hahow.in/@tonykuoyj?tr=tonykuoyj) | <yaojenkuo@ntu.edu.tw>

# ## 目錄
# 
# - 數值（P.3）
# - 文字（P.13）
# - 布林（P.31）
# - 未定義值（P.45）
# - 條件判斷（P.118）
# - 隨堂練習（P.139）

# ## 數值

# ## 基礎資料類別
# 
# - 數值
#     - `int`(integer)
#     - `float`
# - 文字 `str`(string)
# - 布林 `bool`(boolean)
# - 未定義值 `NoneType`

# ## 使用內建函數 `type()` 確認物件所參照的資料類別
# 
# 我們通常都會用一個物件去參照字面值，而物件名稱或 `print()` 函數並不能反映資料類別。

# In[1]:


favorite = "5566" # favorite boy group?
print(favorite)
favorite = 5566   # favorite number?
print(favorite)


# In[2]:


my_lucky_number = 5566
full_marathon_distance_in_km = 42.195
my_favorite_boy_group = "5566"
print(type(my_lucky_number))
print(type(full_marathon_distance_in_km))
print(type(my_favorite_boy_group))


# ## 數值
# 
# - `int`(integer)
# - `float`

# In[3]:


my_lucky_number = 5566
full_marathon_distance_in_km = 42.195
print(type(my_lucky_number))
print(type(full_marathon_distance_in_km))


# ## 數值運算符
# 
# - 直觀的加減乘除 `+`, `-`, `*`, `/`
# - 次方 `**`
# - 計算餘數 `%`
# - 計算商數 `//`
# - 優先運算 `()`

# ## 次方運算的範例：與內建函數 `pow()` 相同作用的自行定義函數 `power()`

# In[4]:


def power(x, n):
    return x**n

print(power(5, 3))
print(pow(5, 3))


# ## 數值運算符的順位
# 
# 1. 最高順位是優先運算 `()`
# 2. 第二順位是次方 `**`
# 3. 第三順位是計算餘數 `%` 計算商數 `//` 乘 `*` 以及除 `/`
# 4. 第四順位是加 `+` 減 `-`

# ## 優先運算的範例：將華氏溫度轉換為攝氏溫度的自行定義函數 `convert_fahrenheit_to_celsius()`
# 
# \begin{equation}
# \text{Celsius}(^{\circ} \text{C}) = (\text{Fahrenheit}(^{\circ} \text{F}) - 32) \times \frac{5}{9}
# \end{equation}

# In[5]:


def convert_fahrenheit_to_celsius(fahrenheit):
    celsius = (fahrenheit - 32) * 5/9  # the formula converts fahrenheit to celsius.
    return celsius

print(convert_fahrenheit_to_celsius(32))
print(convert_fahrenheit_to_celsius(212))


# ## 科學計號 `e`
# 
# - 數值可以支援科學記號（Scientific notation）`e`
# - 注意不要和自然對數函數的底數 $e = 2.71828...$ 搞混了。

# In[6]:


print(3e0)
print(3e2)
print(3e-2)


# ## 文字

# ## 使用成對的單引號 `'`、雙引號 `"` 或三個雙引號 `"""` 形成 `str`

# In[7]:


str_with_single_quotes = 'Hello, world!'
str_with_double_quotes = "Hello, world!"
str_with_triple_double_quotes = """Hello, world!"""
print(type(str_with_single_quotes))
print(type(str_with_double_quotes))
print(type(str_with_triple_double_quotes))


# ## `str` 中可能會包含單引號或雙引號，有兩種解決方式
# 
# 1. 以反斜線 `\`（又稱跳脫符號）作為標註。
# 2. 以不同樣式的引號來形成 `str` 藉此區隔。

# In[8]:


mcd = 'I\'m lovin\' it!'   # escape with \
mcd = "I'm lovin' it!"     # use different quotation marks
mcd = """I'm lovin' it!""" # use different quotation marks


# ## 善用成對的三個雙引號 `"""` 形成 `str`
# 
# - 常用來形成有換行、段落、有單引號以及有雙引號的文章或其他語言的程式碼。
# - 放置在自行定義函數的主體第一列可以作為說明（Docstring）。
# - 利用反斜線 `\` 讓程式碼在顯示換行但作用連續。

# In[9]:


# Use \ for implicit continuation
shawshank_redemption_storyline = """Chronicles the experiences of a formerly successful \
banker as a prisoner in the gloomy jailhouse of Shawshank after being found guilty \
of a crime he did not commit. The film portrays the man's unique way of dealing \
with his new, torturous life; along the way he befriends a number of fellow prisoners, \
most notably a wise long-term inmate named Red."""
type(shawshank_redemption_storyline)


# In[10]:


# Use \ for implicit continuation
sql_query =\
"""
SELECT *
  FROM world
 WHERE country = 'Taiwan';
"""
type(sql_query)


# ## 使用內建函數 `help()` 查詢函數的使用方法

# In[11]:


help(abs)


# ## 將 `str` 放置在自行定義函數主體第一列可以作為說明（Docstring）

# In[12]:


def power(x, n):
    return x**n

help(power)


# In[13]:


def power(x, n):
    """
    Equivalent to x**n.
    """
    return x**n

help(power)


# ## 能對 `str` 使用的文字運算符
# 
# - 加號 `+` 能夠連接 `str`
# - 乘號 `*` 能夠複製 `str` 出現次數。

# In[14]:


m = """I'm"""
c = """ lovin'"""
d = """ it!"""
print(m + c + d)


# In[15]:


mcd = m + c + d
print(mcd*3)


# ## 另一個 `str` 的重要技巧是特定顯示格式
# 
# - 具體來說是在 `str` 中嵌入物件，如此顯示的內容將隨著物件中所儲存的值而變動。
# - 可以透過加號來操作，但是這個做法可讀性較低且無法指定顯示格式。

# In[16]:


def say_hello_to_anyone(anyone):
    return "Hello, " + anyone + "!"

print(say_hello_to_anyone("world"))
print(say_hello_to_anyone("Python"))


# ## 透過 `str.format()` 方法以及 f-string 語法搭配大括號 `{}` 指定顯示格式
# 
# ```python
# "{}".format(object_name) # str.format() 方法
# f"{object_name}"         # f-string 語法
# ```

# ## 特定顯示格式的範例：`say_hello_to_anyone()` 函數

# In[17]:


def say_hello_to_anyone(anyone):
    return "Hello, {}!".format(anyone) # str's format method

print(say_hello_to_anyone("world"))
print(say_hello_to_anyone("Python"))


# In[18]:


def say_hello_to_anyone(anyone):
    return f"Hello, {anyone}!" # f-string syntax

print(say_hello_to_anyone("world"))
print(say_hello_to_anyone("Python"))


# ## 我常用的特定顯示格式
# 
# - 指定小數點 `n` 位的浮點數格式 `:.nf`。
# - 具有千分位逗號的金額格式 `:,`。

# In[19]:


def fahrenheit_to_celsius(fahrenheit):
    celsius = (fahrenheit - 32) * 5/9
    msg = "{:.1f} degrees fahrenheit is equal to {:.1f} degrees celsius.".format(fahrenheit, celsius)
    return msg

print(fahrenheit_to_celsius(32))
print(fahrenheit_to_celsius(212))


# In[20]:


def big_mac_index(country, currency, price):
    msg = f"A Big Mac costs {price:,.2f} {currency} in {country}."
    return msg

print(big_mac_index("Taiwan", "TWD", 72))
print(big_mac_index("US", "USD", 5.65))
print(big_mac_index("South Korea", "Won", 6520))


# ## 其他更多特定顯示格式
# 
# <https://www.w3schools.com/python/ref_string_format.asp>

# ## 布林

# ## 什麼是 `bool`
# 
# 布林（Boolean）是程式設計中用來表示邏輯的資料類別，以發明布林代數的數學家 George Boole 為名，它只有兩種值：假（`False`）和真（`True`），在程式設計與資料科學可以進行流程控制以及資料篩選。

# ## 有三種方式可以形成 `bool`
# 
# 1. 直接使用保留字 `False` 或 `True`
# 2. 使用關係運算符。
# 3. 使用邏輯運算符。

# ## 直接使用保留字 `False` 或 `True`

# In[21]:


bool_false = False
bool_true = True

print(bool_false)
print(bool_true)
print(type(bool_false))
print(type(bool_true))


# ## 使用關係運算符形成 `bool`
# 
# - 等於 `==`
# - 不等於 `!=`
# - 大於 `>` 小於 `<`
# - 大於等於 `>=` 小於等於 `<=`
# - 包含於 `in` 不包含於 `not in`

# ## 在關係運算符兩側放置物件或字面值形成 `bool` 稱為條件運算式（Conditional expression）

# In[22]:


my_lucky_number = 5566
my_favorite_boy_group = "5566"

print(my_lucky_number == 5566)
print(my_lucky_number != 5566)
print(my_lucky_number > 5566)
print(my_lucky_number >= 5566)
print('56' in my_favorite_boy_group)
print('78' not in my_favorite_boy_group)


# ## 要將 `==` 以及 `=` 明確地區分出來
# 
# - `==` 是關係運算符，用來判斷符號兩側是否相等。
# - `=` 是賦值運算符，用來讓左側的物件名稱成為右側類別的實例。

# In[23]:


my_favorite_boy_group = "5566"
my_favorite_boy_group == 5566


# ## 使用邏輯運算符形成 `bool`
# 
# - 交集 `and`
# - 聯集 `or`
# - 非 `not`

# ## 非 `not` 運算符將 `bool` 反轉

# In[24]:


bool_false = False
bool_true = True
print(not bool_false)
print(not bool_true)


# ## 邏輯運算符針對已經是 `bool` 的資料進行集合運算
# 
# - 在交集 `and` 運算符兩側必須都是 `True` 判斷結果才會是 `True`
# - 在聯集 `or` 運算符兩側必須都是 `False` 判斷結果才會是 `False`

# ## 在交集 `and` 運算符兩側必須都是 `True` 判斷結果才會是 `True`

# In[25]:


bool_false = False
bool_true = True
print(bool_true and bool_true)
print(bool_true and bool_false)
print(bool_false and bool_true)
print(bool_false and bool_false)


# ## 在聯集 `or` 運算符兩側必須都是 `False` 判斷結果才會是 `False`

# In[26]:


bool_false = False
bool_true = True
print(bool_false or bool_false)
print(bool_true or bool_true)
print(bool_true or bool_false)
print(bool_false or bool_true)


# ## 交集 `and` 運算符的範例：`is_a_good_marathon_weather()` 函數
# 
# 如果一個適合跑馬拉松的天氣定義是「乾」**並且**「冷」，假設比賽日乾燥與低溫的機率均為 50%，適合跑馬拉松的機率為 25%。

# In[27]:


def is_a_good_marathon_weather(is_dry, is_cold):
    return is_dry and is_cold

print(is_a_good_marathon_weather(True, True))
print(is_a_good_marathon_weather(True, False))
print(is_a_good_marathon_weather(False, True))
print(is_a_good_marathon_weather(False, False))


# ## 聯集 `or` 運算符的範例：`is_a_good_marathon_weather()` 函數
# 
# 如果一個適合跑馬拉松的天氣定義是「乾」**或者**「冷」，假設比賽日乾燥與低溫的機率均為 50%，適合跑馬拉松的機率為 75%。

# In[28]:


def is_a_good_marathon_weather(is_dry, is_cold):
    return is_dry or is_cold

print(is_a_good_marathon_weather(False, False))
print(is_a_good_marathon_weather(True, True))
print(is_a_good_marathon_weather(True, False))
print(is_a_good_marathon_weather(False, True))


# ## 未定義值

# ## 什麼是 `NoneType`
# 
# - Python 表示未定義值、空值或者虛無值的特殊資料類別，只有一個值：`None`。
# - `NoneType` 既不是 `False`。
# - `NoneType` 也不是空的 `str` 類別 `''`。
# - `NoneType` 也不是 `0`。

# In[29]:


a_none_type = None
print(type(a_none_type))
print(a_none_type == None)
print(a_none_type == False)
print(a_none_type == '')
print(a_none_type == 0)


# ## 如果在自行定義函數時沒有使用 `return`，函數將輸出 `NoneType`
# 
# 這也是「練習題指引」中寫到若只是用 `print()` 函數將預期輸出印出，並無法通過測試的根本原因，輸出 `NoneType` 無法通過和預期輸出比對的測試資料。

# In[30]:


def convert_fahrenheit_to_celsius(fahrenheit):
    celsius = (fahrenheit - 32) * 5/9
    print(celsius) # Instead of return celsius, we just print celsius.
    
function_output = convert_fahrenheit_to_celsius(212)
print(function_output)
print(type(function_output))


# ## 資料類別判斷的兩種方式
# 
# 1. 將 `type()` 函數的輸出與資料類別名稱比較。
# 2. 透過內建函數 `isinstance()` 判斷。

# ## 資料類別可以透過內建函數 `type()` 判斷
# 
# 將 `type()` 函數的輸出與資料類別名稱比較。

# In[31]:


a_bool_false = False
a_str_false = "False"
print(type(a_bool_false) == bool)
print(type(a_str_false) == bool)


# In[32]:


an_integer_5566 = 5566
a_str_5566 = "5566"
print(type(an_integer_5566) == int)
print(type(a_str_5566) == int)


# In[33]:


a_none_type = None
a_none_str = "None"
print(type(a_none_type) == type(None))
print(type(a_none_str) == type(None))


# ## 資料類別可以透過內建函數 `isinstance()` 判斷
# 
# `isinstance(x, classinfo)` 函數判斷輸入物件 `x` 是否為某個資料類別的實例，其中 `classinfo` 參數輸入欲判斷的資料類別名稱。

# In[34]:


# Use bool as classinfo
a_bool_false = False
a_str_false = "False"
print(isinstance(a_bool_false, bool))
print(isinstance(a_str_false, bool))


# In[35]:


# Use int as classinfo
an_integer_5566 = 5566
a_str_5566 = "5566"
print(isinstance(an_integer_5566, int))
print(isinstance(a_str_5566, int))


# In[36]:


# Use type(None) as classinfo
a_none_type = None
a_none_str = "None"
print(isinstance(a_none_type, type(None)))
print(isinstance(a_none_str, type(None)))


# ## 資料類別可以透過與其同名的內建函數轉換
# 
# - `bool()` 可以將輸入資料轉換為 `bool`
# - `int()` 可以將輸入資料轉換為 `int`
# - `float()` 可以將輸入資料轉換為 `float`
# - `str()` 可以將輸入資料轉換為 `str`

# ## 資料類別轉換的包容性
# 
# 由資料類別的子集合往資料類別的母集合轉換可以確保不會出錯。
# 
# \begin{equation}
# \text{bool} \in \text{int} \in \text{float} \in \text{str}
# \end{equation}

# In[37]:


# Upcasting is always allowed.
print(int(False))
print(float(0))
print(str(0.0))


# ## 資料類別轉換的包容性（續）
# 
# 由資料類別的母集合往資料類別的子集合轉換需要注意是否符合邏輯、是否正確。
# 
# \begin{equation}
# \text{bool} \in \text{int} \in \text{float} \in \text{str}
# \end{equation}

# In[38]:


print(bool('0'))
print(bool('False'))


# ## `str` 無法被轉換為 `float`
# 
# `try...except...` 是例外處理的語法，之後在「使用流程控制管理程式區塊的執行」章節會說明。

# In[39]:


try:
    print(float('Hello world!'))
except ValueError as error_message:
    print(error_message)


# ## 為什麼需要資料結構
# 
# - 在資料科學家日常的工作任務中，資料處理佔有相當高的比例。
# - 需要有一個機制能夠協助他們輸入、處理最後輸出資料。
# - 這個「機制」就是**資料結構**。
# - 適當地選擇資料結構，讓資料科學家能夠有效率地儲存與取得資料。
# - 就像是將食物放置在冰箱、衣服放置在衣櫥、鞋子放置在鞋櫃。

# ## Python 內建的四個資料結構類別
# 
# 1. `list`
# 2. `tuple`
# 3. `dict`(dictionary)
# 4. `set`

# ## `list`
# 
# - `list` 是一種「有序」且能夠「更新」的資料結構。
# - `list` 可以透過「逗號」`,` 分隔值與「中括號」`[]` 形成。

# ## 命名物件作為 `list` 類別的實例

# In[40]:


primes = [2, 3, 5, 7, 11]
type(primes)


# ## 使用內建函數 `len()`  得知一個 `list` 中有幾個資料值

# In[41]:


len(primes)


# ## 從一個 `list` 中取出資料的方式有兩種
# 
# 1. **indexing** 指的是從一個 `list` 中取出特定位置的單個資料值。
# 2. **slicing** 指的是從一個 `list` 中擷取特定片段。

# ## indexing `[index]`
# 
# `list` 採用兩個方向的索引機制：
# 
# 1. 由左至右：「從 0 開始」的索引機制。
# 2. 由右至左：「從 -1 開始」的索引機制。

# In[42]:


print(primes[0])  # the first element
print(primes[1])  # the second element
print(primes[-1]) # the last element
print(primes[-2]) # the second last element


# ## slicing `[start:stop:step]`
# 
# 除了可以取出特定位置的單個資料值，`list` 也支援擷取特定片段，藉此獲得一個較短長度 `list` 的語法。
# 
# - `start` 起始位置（包含）。
# - `stop` 終止位置（排除）。
# - `step` 間隔。

# In[43]:


print(primes[0:3:1])            # slicing the first 3 elements
print(primes[-3:len(primes):1]) # slicing the last 3 elements 
print(primes[0:len(primes):2])  # slicing every second element


# ## slicing 如果沒有指定 `start:stop:step` 就採用預設值
# 
# - `start` 起始位置（包含）預設為 `0`。
# - `stop` 終止位置（排除）預設為 `list` 的長度。
# - `step` 間隔預設為 `1`。

# In[44]:


print(primes[:3])  # slicing the first 3 elements
print(primes[-3:]) # slicing the last 3 elements 
print(primes[::2]) # slicing every second element


# ## `str` 也適用 indexing 與 slicing

# In[45]:


luke_skywalker = "Luke Skywalker"
print(luke_skywalker[0])
print(luke_skywalker[-1])
print(luke_skywalker[:4])
print(luke_skywalker[5:])


# ## 更新 `list` 中的資料值

# In[46]:


print(primes)   # before update
primes[-1] = 13 # update
print(primes)   # after update


# ## 也可以透過 `list` 的多種方法更新
# 
# - `list.append()`
# - `list.pop()`
# - `list.remove()`
# - `list.insert()`
# - `list.sort()`
# - ...等。

# ## 使用 `help()` 函數查詢 `list` 方法的說明

# In[47]:


help(primes.append)


# ## 在小括號中按 `Shift - Tab` 查詢方法的說明

# In[48]:


#primes.append()


# ## `list` 支援的運算符
# 
# - `+` 運算符：連接 lists
# - `*` 運算符：複製 `list` 中的元素。

# In[49]:


primes = [2, 3, 5, 7, 11]
primes_to_concatenate = [13, 17, 19, 23, 29]
primes + primes_to_concatenate


# In[50]:


print(primes * 2)
print(primes_to_concatenate * 3)


# ## 在應用 `list` 的多種方法之前，先注意函數的兩種使用形式
# 
# 1. 對物件應用函數，語法為 `function(object)`
# 2. 使用附屬於物件的函數，稱為使用物件的方法，語法為 `object.method()`

# ## 除了使用函數與方法的差異，也要留意物件更新的兩種方式
# 
# 1. 將更新的結果回傳（Return），物件本身維持不變。
# 2. 更新物件本身並回傳 `None`

# ## 兩種更新方式的範例：排序一個 `list`
# 
# 1. 使用內建函數 `sorted()` 採取將更新的結果回傳（Return），物件本身維持不變的更新方式。
# 2. 使用 `list.sort()` 採取更新物件本身並回傳 `None` 的更新方式。

# In[51]:


unsorted_primes = [11, 5, 7, 2, 3]
sorted_primes = sorted(unsorted_primes)
print(unsorted_primes)
print(sorted_primes)


# In[52]:


unsorted_primes = [11, 5, 7, 2, 3]
sorted_primes = unsorted_primes.sort()
print(unsorted_primes)
print(sorted_primes)


# ## 如何判斷兩種更新方式
# 
# - 詳細閱讀函數與方法的文件。
# - 注意函數與方法使用後儲存格是否有 Out 顯示。

# ## 詳細閱讀函數與方法的說明
# 
# 內建函數 `sorted()`：**Return** a new list containing all items from the iterable in ascending order.

# In[53]:


help(sorted)


# ## 詳細閱讀函數與方法的說明（續）
# 
# `list.sort()`：Sort the list in ascending order and **return None**.

# In[54]:


help(unsorted_primes.sort)


# ## 注意函數與方法使用後儲存格是否有 Out 顯示

# In[55]:


unsorted_primes = [11, 5, 7, 2, 3]
sorted(unsorted_primes)


# In[56]:


unsorted_primes.sort()


# ## 兩種更新方式並不一定是「擇一」也可能都有

# In[57]:


primes = [2, 3, 5, 7, 11]
the_last_element = primes.pop()
print(primes)
print(the_last_element)


# ## 釐清兩種更新方式是至關重要的
# 
# - 雖然是利用 `list` 作為範例，但不管任何的資料或者資料結構都適用。
# - 多數的函數更新機制都是回傳（Return），但並不是絕對。
# - 不論函數或者方法，都要詳細閱讀文件或注意使用後儲存格是否有 Out 顯示。

# ## `tuple`
# 
# - `tuple` 是一種「有序」且「不能夠更新」的資料結構。
# - `tuple` 可以透過「逗號」`,` 分隔值與「小括號」`()` 形成。

# ## 命名物件作為 `tuple` 類別的實例

# In[58]:


primes = (2, 3, 5, 7, 11)
type(primes)


# ## 使用內建函數 `len()`  得知一個 `tuple` 中有幾個資料值

# In[59]:


len(primes)


# ## `tuple` 在許多地方的表現與 `list` 相同，像是 indexing 以及 slicing

# In[60]:


print(primes[0])   # the first element
print(primes[:3])  # slicing the first 3 elements


# ## `tuple` 與 `list` 最大的不同點，在於 `tuple`「不能夠更新」的特性
# 
# 以更新 `list` 的語法更新 `tuple` 會產生錯誤。

# In[61]:


try:
    primes[-1] = 13
except TypeError as error_message:
    print(error_message)


# ## `dict`
# 
# - `dict` 是一種使用「鍵值對應」關係的資料結構。
# - `dict` 可以透過「逗號」`,`、「鍵值對應」`key: value` 與「大括號」`{}` 形成。

# In[62]:


the_shawshank_redemption = {
    "title": "The Shawshank Redemption",
    "year": 1995,
    "rating": 9.3,
    "director": "Frank Darabont"
}
type(the_shawshank_redemption)


# ## 使用內建函數 `len()`  得知一個 `dict` 中有幾組鍵值對應

# In[63]:


len(the_shawshank_redemption)


# ## `dict` 採用以「鍵」取「值」的索引機制
# 
# ```python
# dict["key"]
# ```

# In[64]:


print(the_shawshank_redemption["title"])
print(the_shawshank_redemption["year"])
print(the_shawshank_redemption["rating"])
print(the_shawshank_redemption["director"])


# ## 用來檢視 `dict` 的三個方法
# 
# 1. `dict.keys()` 檢視鍵。
# 2. `dict.values()` 檢視值。
# 3. `dict.items()` 檢視鍵值對應。

# In[65]:


print(the_shawshank_redemption.keys())
print(the_shawshank_redemption.values())
print(the_shawshank_redemption.items())  # (key, value) as a tuple


# ## 新增 `dict` 中的鍵值對應

# In[66]:


the_shawshank_redemption['lead_actors'] = ['Tim Robbins', 'Morgan Freeman']
the_shawshank_redemption


# ## 使用 `del` 保留字刪除 `dict` 中的鍵值對應

# In[67]:


del the_shawshank_redemption['lead_actors']
the_shawshank_redemption


# ## 使用 `dict.pop(key)` 方法指定「鍵」將「值」拋出

# In[68]:


the_shawshank_redemption.pop("director")


# ## 指定 `dict` 的「鍵」更新「值」

# In[69]:


the_shawshank_redemption["year"] = 1994
the_shawshank_redemption


# ## `set`
# 
# - `set` 是一種「無序」、儲存「獨一值」並且能夠進行「集合運算」的資料結構。
# - `set` 可以透過「逗號」`,` 分隔值與「大括號」`{}` 形成。

# In[70]:


primes = {2, 3, 5, 7, 7}  # 7 is duplicated
odds = {1, 3, 5, 7, 9, 9} # 9 is duplicated
print(type(primes))
print(type(odds))


# ## 使用內建函數 `len()`  得知一個 `set` 中有幾個獨一值

# In[71]:


print(len(primes))
print(primes)
print(len(odds))
print(odds)


# ## `set` 重要的兩個特性
# 
# 1. 不支援 indexing
# 2. 支援集合運算（Set operations）。

# ## `set` 不支援 indexing

# In[72]:


try:
    primes[0]
except TypeError as error_message:
    print(error_message)


# ## `set` 支援集合運算
# 
# - 使用集合運算符（Set operators）。
# - 使用 `set` 類別的方法。

# ## 常用的集合運算
# 
# - 交集。
# - 聯集。
# - 差集。
# - 對稱差集。

# ## 使用 `set.intersection()` 方法

# In[73]:


print(primes.intersection(odds))


# ## 使用 `set.union()` 方法

# In[74]:


print(primes.union(odds))


# ## 使用 `set.difference()` 方法

# In[75]:


print(primes.difference(odds))
print(odds.difference(primes))


# ## 使用 `set.symmetric_difference()` 方法

# In[76]:


print(primes.symmetric_difference(odds))


# ## 截至目前為止，我們已經看過了三種括號的應用場景
# 
# 1. 小括號 `()`
# 2. 中括號 `[]`
# 3. 大括號 `{}`

# ## 小括號 `()` 的應用場景
# 
# - 對物件應用函數：`function(object)`
# - 運算的優先順序。
# - 使用物件的方法：`object.method()`
# - 形成 `tuple` 資料結構類別。

# ## 中括號 `[]` 的應用場景
# 
# - 形成 `list` 資料結構類別。
# - 從不同資料結構類別 indexing/slicing 資料。

# ## 大括號 `{}` 的應用場景
# 
# - 與 `str.format()` 以及 f-string 語法搭配。
# - 形成 `dict` 資料結構類別。
# - 形成 `set` 資料結構類別。

# ## Python 資料結構類別具備有威力的「複合性」
# 
# 複合性指的是資料結構類別中能夠包含異質的資料類別以及不同的資料結構類別，在資料處理上相當有優勢。
# 
# - 資料類別：`int`/`float`/`str`/`bool`/`NoneType`
# - 資料結構類別：`list`/`tuple`/`dict`/`set` 

# ## 條件判斷

# ## 流程控制
# 
# 多數程式語言都會從程式碼的第一列開始按照列（Row-wise）的順序往下讀取並且執行，但是在某些情況下，我們會希望依據特定的條件來決定程式的執行與否、重複次數以及錯誤發生時該如何應對，這時就可以透過流程控制的結構機制來滿足這些情況。

# ## 我們將要學習的流程控制
# 
# - 條件判斷。
# - 迴圈。

# ## 什麼是程式區塊
# 
# > 程式區塊（Code block）有時也被稱為複合語句，是將程式組合並產生依附關係的結構，由一個或多個敘述所組成。
# 
# 來源：<https://en.wikipedia.org/wiki/Block_(programming)>

# ## Python 使用四個空白作為縮排（Indentation）標註程式區塊
# 
# - 多數程式語言使用大括號 `{}` 來標註程式碼所依附的特定保留字。
# - 一段程式碼的依附關係從縮排開始直到第一個未縮排的結束。
# - 縮排必須隨著依附保留字的數量而增加。

# ## 什麼時候需要用到程式區塊
# 
# - 流程控制。
# - 定義函數與類別。

# ## 使用「條件」與「縮排」建立條件敘述
# 
# - 條件指的是一段能夠被解讀為 `bool` 的敘述。
# - 縮排是 Python 用來辨識程式碼依附區塊的結構，要特別留意。

# ## 使用 `if` 依據條件決定是否執行程式區塊
# 
# ```python
# if 條件:
#     # 依附 if 敘述的程式區塊。
#     # 當條件為 True 的時候程式區塊才會被執行。
# ```

# ## 使用關係運算符或者邏輯運算符描述條件
# 
# - 關係運算符：`==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `not in`
# - 邏輯運算符：`and`, `or`, `not`

# In[77]:


def return_message_if_positive(x):
    if x > 0:
        return f"{x} is positive."

print(return_message_if_positive(56))
print(return_message_if_positive(-56))
print(return_message_if_positive(0))


# ## 使用 `if...else...` 依據條件決定執行兩個程式區塊其中的一個
# 
# ```python
# if 條件:
#     # 依附 if 敘述的程式區塊。
#     # 當條件為 True 的時候會被執行。
# else:
#     # 依附 else 敘述的程式區塊。
#     # 當條件為 False 的時候會被執行。
# ```

# In[78]:


def return_message_whether_positive_or_not(x):
    if x > 0:
        return f"{x} is positive."
    else:
        return f"{x} is not positive."

print(return_message_whether_positive_or_not(56))
print(return_message_whether_positive_or_not(0))
print(return_message_whether_positive_or_not(-56))


# ## 使用 `if...elif...else...` 依據條件決定執行多個程式區塊其中的一個
# 
# ```python
# if 條件一:
#     # 依附 if 敘述的程式區塊。
#     # 當條件一為 True 的時候會被執行。
# elif 條件二:
#     # 依附 elif 敘述的程式區塊。
#     # 當條件一為 False 、條件二為 True 的時候會被執行。
# else:
#     # 依附 else 敘述的程式區塊。
#     # 當條件一、條件二均為 False 的時候會被執行。
# ```

# In[79]:


def return_message_whether_positive_negative_or_neutral(x):
    if x > 0:
        return f"{x} is positive."
    elif x < 0:
        return f"{x} is negative."
    else:
        return f"{x} is neutral."

print(return_message_whether_positive_negative_or_neutral(56))
print(return_message_whether_positive_negative_or_neutral(-56))
print(return_message_whether_positive_negative_or_neutral(0))


# ## 使用 `if...elif...` 把所有的條件都寫清楚
# 
# 不一定非要加入 `else`

# In[80]:


def return_message_whether_positive_negative_or_neutral(x):
    if x > 0:
        return f"{x} is positive."
    elif x < 0:
        return f"{x} is negative."
    elif x == 0:
        return f"{x} is neutral."

print(return_message_whether_positive_negative_or_neutral(56))
print(return_message_whether_positive_negative_or_neutral(-56))
print(return_message_whether_positive_negative_or_neutral(0))


# ## 一組條件敘述的結構僅會執行「其中一個」程式區塊
# 
# - 如果條件彼此之間**互斥**，寫作條件的先後順序**沒有**影響。
# - 如果條件彼此之間**非互斥**，寫作條件的先後順序**有**影響。

# ## 以 Fizz buzz 為例
# 
# > 從 1 數到 100，碰到 3 的倍數改為 Fizz、碰到 5 的倍數改為 Buzz，碰到 15 的倍數改為 Fizz Buzz，其餘情況不改動。
# 
# 來源：<https://en.wikipedia.org/wiki/Fizz_buzz>

# ## Fizz buzz 值得注意的地方
# 
# 條件彼此之間**非**互斥（15 是 3 與 5 的公倍數），寫作條件的先後順序**有**影響。

# ## 使用 `if...elif...` 定義 `fizz_buzz()` 函數

# In[81]:


def fizz_buzz(x):
    if x % 3 != 0 and x % 5 != 0 and x % 15 != 0:
        return x
    elif x % 15 == 0:
        return "Fizz Buzz"
    elif x % 3 == 0:
        return "Fizz"
    elif x % 5 == 0:
        return "Buzz"

print(fizz_buzz(2))
print(fizz_buzz(3))
print(fizz_buzz(5))
print(fizz_buzz(15))


# ## 使用 `if...elif...else...` 定義 `fizz_buzz()` 函數

# In[82]:


def fizz_buzz(x):
    if x % 15 == 0:
        return "Fizz Buzz"
    elif x % 3 == 0:
        return "Fizz"
    elif x % 5 == 0:
        return "Buzz"
    else:
        return x

print(fizz_buzz(2))
print(fizz_buzz(3))
print(fizz_buzz(5))
print(fizz_buzz(15))


# ## 假如在寫作條件敘述時不想要去考慮條件的先後順序
# 
# 那就要記得把條件描述為**互斥**。

# In[83]:


def fizz_buzz(x):
    if x % 3 == 0 and x % 15 != 0:
        return "Fizz"
    elif x % 5 == 0 and x % 15 != 0:
        return "Buzz"
    elif x % 15 == 0:
        return "Fizz Buzz"
    else:
        return x

print(fizz_buzz(2))
print(fizz_buzz(3))
print(fizz_buzz(5))
print(fizz_buzz(15))


# ## 隨堂練習

# ## 隨堂練習
# 
# <https://colab.research.google.com/github/datainpoint/classroom-fintech-bot-2026/blob/main/02-exercises.ipynb>
