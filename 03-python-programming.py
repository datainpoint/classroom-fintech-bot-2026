#!/usr/bin/env python
# coding: utf-8

# # 臺灣銀行委辦 AI 人才進階訓練
# 
# > Python 程式設計（三），2026-04-02
# 
# [郭耀仁](https://hahow.in/@tonykuoyj?tr=tonykuoyj) | <yaojenkuo@ntu.edu.tw>

# ## 目錄
# 
# - 函數（P.3）
# - 彈性參數（P.53）
# - 作用域（P.64）
# - 類別（P.81）
# - 模組（P.98）
# - 隨堂練習（P.114）

# ## 函數

# ## 重複的任務
# 
# - 迴圈：重複執行程式區塊的程式碼。
# - 函數：為程式區塊的程式碼命名，進而方便重複使用。

# ## 什麼是迴圈
# 
# > 迴圈是流程控制的其中一種技巧，可以讓寫作一次的程式區塊被重複執行，常見的應用是重複執行直到條件不成立時或走訪可迭代類別中的所有元素。
# 
# 來源：<https://en.wikipedia.org/wiki/Control_flow#Loops>

# ## 迴圈的三個要素
# 
# 1. 起始。
# 2. 終止。
# 3. 如何從起始到終止。

# ## 兩種常見的迴圈
# 
# 1. `while` 迴圈：重複執行程式區塊直到條件為 `False` 的時候。
# 2. `for` 迴圈：走訪可迭代類別中的所有元素。

# ## 使用 `while` 依據條件決定是否重複執行程式區塊
# 
# ```python
# while 條件:
#     # 依附 while 敘述的程式區塊。
#     # 當條件為 True 的時候程式區塊會被重複執行。
#     # 當條件為 False 的時候停止執行程式區塊。
# ```

# ## 如何寫作一個 `while` 迴圈
# 
# - 在迴圈程式區塊之前定義一個物件設定起始值。
# - 設計條件讓程式區塊重複執行的次數符合我們的需求。
# - 記得在程式區塊中更新物件的值。

# ## 如何寫作一個 `while` 迴圈：印出 5 次 `"Hello, world!"`
# 
# 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=number_of_prints%20%3D%200%0Awhile%20number_of_prints%20%3C%205%3A%0A%20%20%20%20print%28%22Hello,%20world!%22%29%0A%20%20%20%20number_of_prints%20%3D%20number_of_prints%20%2B%201&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 拆解程式執行的每個步驟。

# In[1]:


number_of_prints = 0
while number_of_prints < 5:
    print("Hello, world!")
    number_of_prints = number_of_prints + 1


# ## 在程式區塊中更新物件的值更常會使用複合運算符（Compound operators）
# 
# - `integer += 1` 等同於 `integer = integer + 1` 
# - `integer -= 1` 等同於 `integer = integer - 1` 
# - `integer *= 1` 等同於 `integer = integer * 1` 
# - `integer /= 1` 等同於 `integer = integer / 1` 
# - ...等。

# ## 如何寫作一個 `while` 迴圈：印出小於 10 的奇數
# 
# 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=odd%20%3D%201%0Awhile%20odd%20%3C%2010%3A%0A%20%20%20%20print%28odd%29%0A%20%20%20%20odd%20%2B%3D%202%20%23%20odd%20%3D%20odd%20%2B%202&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 拆解程式執行的每個步驟。

# In[2]:


odd = 1
while odd < 10:
    print(odd)
    odd += 2 # odd = odd + 2


# ## 如何寫作一個 `while` 迴圈：從週一印到週五
# 
# 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=weekdays%20%3D%20%5B%22Monday%22,%20%22Tuesday%22,%20%22Wednesday%22,%20%22Thursday%22,%20%22Friday%22%5D%0Aindex%20%3D%200%0Awhile%20index%20%3C%20len%28weekdays%29%3A%0A%20%20%20%20print%28weekdays%5Bindex%5D%29%0A%20%20%20%20index%20%2B%3D%201&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 拆解程式執行的每個步驟。

# In[3]:


weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
index = 0
while index < len(weekdays):
    print(weekdays[index])
    index += 1


# ## 使用 `for` 走訪可迭代類別（Iterables）中的所有元素
# 
# ```python
# for 元素 in 可迭代類別:
#     # 依附 for 敘述的程式區塊。
#     # 當可迭代類別還沒有走訪完的時候程式區塊會被重複執行。
#     # 當可迭代類別走訪完的時候停止執行程式區塊。
# ```

# ## 什麼是可迭代類別
# 
# 具有一次回傳其中一個資料值特性的類別、輸入到內建函數 `iter()` 不會產生錯誤的類別，都屬於可迭代類別（Iterables），常見的有 `str` 與資料結構。
# 
# - 資料類別：`str`
# - 資料結構類別：`list`、`tuple`、`dict`、`set`

# In[4]:


luke = "Luke Skywalker"
primes = [2, 3, 5, 7, 11]
iter(luke)
iter(primes)


# ## 什麼是不可迭代的類別
# 
# 任何輸入到內建函數 `iter()` 會產生錯誤的類別都是不可迭代類別，像是 `int`、`float` 與 `bool` 等。

# In[5]:


i_am_int = 5566
try:
    iter(i_am_int)
except TypeError as error_message:
    print(error_message)


# In[6]:


i_am_float = 5566.0
try:
    iter(i_am_float)
except TypeError as error_message:
    print(error_message)


# In[7]:


i_am_bool = False
try:
    iter(i_am_bool)
except TypeError as error_message:
    print(error_message)


# ## 如何寫作一個 `for` 迴圈
# 
# - 建立一個可迭代類別。
# - 可迭代類別如果是數列，可透過內建函數 `range()` 建立。

# ## `range()` 函數有三個參數可以設定數列內容
# 
# 1. `start` 數列的起始整數，即第 0 個整數（包含），預設值為 0。
# 2. `stop` 數列的終止整數，即第 -1 個整數（排除）。
# 3. `step` 數列的公差，預設值為 1。

# In[8]:


help(range)


# ## 如何寫作一個 `for` 迴圈：印出 5 次 `"Hello, world!"`
# 
# 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=for%20element%20in%20range%280,%205,%201%29%3A%0A%20%20%20%20print%28%22Hello,%20world!%22%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 拆解程式執行的每個步驟。

# In[9]:


for _ in range(0, 5, 1):
    print("Hello, world!")


# ## 如何寫作一個 `for` 迴圈：印出小於 10 的奇數
# 
# 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=for%20odd%20in%20range%281,%2010,%202%29%3A%0A%20%20%20%20print%28odd%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 拆解程式執行的每個步驟。

# In[10]:


for odd in range(1, 10, 2):
    print(odd)


# ## 如何寫作一個 `for` 迴圈：從週一印到週五
# 
# 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=weekdays%20%3D%20%5B%22Monday%22,%20%22Tuesday%22,%20%22Wednesday%22,%20%22Thursday%22,%20%22Friday%22%5D%0Afor%20weekday%20in%20weekdays%3A%0A%20%20%20%20print%28weekday%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 拆解程式執行的每個步驟。

# In[11]:


weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
for weekday in weekdays:
    print(weekday)


# ## 如何抉擇使用哪種迴圈，`for` 迴圈或 `while` 迴圈
# 
# - 先思考問題是否能夠建立可迭代類別？
# - 如果可以，代表程式區塊被重複執行的次數**已知**，選擇 `for` 迴圈。
# - 如果不可以，代表程式區塊被重複執行的次數**未知**，選擇 `while` 迴圈。

# ## 常見的迴圈應用
# 
# - 走訪 `str` 或資料結構。
# - 可迭代類別的加總、乘積與計數。
# - 合併資料成為 `str`、`list` 或者 `dict`。

# ## 走訪 `str` 或資料結構
# 
# 1. 走訪 `str`、`list`、`tuple`、`set`
# 2. 走訪 `dict`

# ## 如何走訪 `str`、`list`、`tuple`、`set`

# In[12]:


def iterate_str_list_tuple_set(an_iterable):
    for element in an_iterable:
        print(element)
        
luke = "Luke Skywalker"
weekdays_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
weekdays_tuple = tuple(weekdays_list)
weekdays_set = set(weekdays_list)


# In[13]:


iterate_str_list_tuple_set(luke) # iterate over a str


# In[14]:


iterate_str_list_tuple_set(weekdays_list)  # iterate over a list


# In[15]:


iterate_str_list_tuple_set(weekdays_tuple) # iterate over a tuple


# In[16]:


iterate_str_list_tuple_set(weekdays_set)   # iterate over a set


# ## 如何走訪 `dict`
# 
# 善用三個 `dict` 方法：
# 
# 1. `dict.keys()`
# 2. `dict.values()`
# 3. `dict.items()`

# In[17]:


the_shawshank_redemption = {
    'title': 'The Shawshank Redemption',
    'year': 1994,
    'rating': 9.3,
    'director': 'Frank Darabont'
}
type(the_shawshank_redemption)


# ## 預設走訪 `dict` 的「鍵」

# In[18]:


for key in the_shawshank_redemption:
    print(key)


# In[19]:


for k in the_shawshank_redemption.keys():
    print(k)


# ## 指定走訪 `dict` 的「值」

# In[20]:


for value in the_shawshank_redemption.values():
    print(value)


# ## 同時走訪 `dict` 的「鍵」與「值」

# In[21]:


dict_items = the_shawshank_redemption.items()
print(dict_items)
for key, value in dict_items:
    print(f"{key}: {value}")


# ## 可迭代類別的加總、乘積與計數
# 
# - 在迴圈程式區塊之前定義一個物件設定起始值。
# - 在程式區塊中更新物件的值。
# 
# 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=summation%20%3D%200%0Aproduct%20%3D%201%0Acount%20%3D%200%0Aprimes%20%3D%20%5B2,%203,%205,%207,%2011%5D%0Afor%20prime%20in%20primes%3A%0A%20%20%20%20summation%20%2B%3D%20prime%0A%20%20%20%20product%20*%3D%20prime%0A%20%20%20%20count%20%2B%3D%201%0Aprint%28summation%29%0Aprint%28product%29%0Aprint%28count%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 拆解程式執行的每個步驟。

# In[22]:


summation = 0
product = 1
count = 0
primes = [2, 3, 5, 7, 11]
for prime in primes:
    summation += prime
    product *= prime
    count += 1
print(summation)
print(product)
print(count)


# ## 可迭代類別的加總與計數
# 
# 善用內建函數 `sum()` 以及 `len()` 就可以得知加總與計數。

# In[23]:


print(sum(primes))
print(len(primes))


# ## 合併資料成為 `str`、`list` 或者 `dict`
# 
# - 運用 `+` 運算符連接元素成為 `str`
# - 運用 `+` 運算符連接 lists
# - 運用 `list.append()` 方法合併元素成為 `list`
# - 運用 `dict[key]=value` 合併元素成為 `dict`

# ## 運用 `+` 運算符連接元素成為 `str`
# 
# 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=vowels%20%3D%20%5B%22a%22,%20%22e%22,%20%22i%22,%20%22o%22,%20%22u%22,%20%22A%22,%20%22E%22,%20%22I%22,%20%22O%22,%20%22U%22%5D%0Avowels_str%20%3D%20str%28%29%20%23%20an%20empty%20str%0Afor%20vowel%20in%20vowels%3A%0A%20%20%20%20vowels_str%20%2B%3D%20vowel%0Aprint%28vowels_str%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 拆解程式執行的每個步驟。

# In[24]:


vowels = ["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"]
vowels_str = str() # an empty str
for vowel in vowels:
    vowels_str += vowel
print(vowels_str)


# ## 運用 `+` 運算符連接 lists
# 
# 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=vowels%20%3D%20%5B%5B%22a%22,%20%22e%22,%20%22i%22,%20%22o%22,%20%22u%22%5D,%20%5B%22A%22,%20%22E%22,%20%22I%22,%20%22O%22,%20%22U%22%5D%5D%0Aflat_vowels%20%3D%20list%28%29%0Afor%20vowel%20in%20vowels%3A%0A%20%20%20%20flat_vowels%20%2B%3D%20vowel%0Aflat_vowels&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 拆解程式執行的每個步驟。

# In[25]:


vowels = [["a", "e", "i", "o", "u"], ["A", "E", "I", "O", "U"]]
flat_vowels = list()
for vowel in vowels:
    flat_vowels += vowel
flat_vowels


# ## 運用 `list.append()` 方法合併元素成為 `list`
# 
# 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=vowels%20%3D%20%5B%5B%22a%22,%20%22e%22,%20%22i%22,%20%22o%22,%20%22u%22%5D,%20%5B%22A%22,%20%22E%22,%20%22I%22,%20%22O%22,%20%22U%22%5D%5D%0Aflat_vowels%20%3D%20list%28%29%0Afor%20v_list%20in%20vowels%3A%0A%20%20%20%20for%20v%20in%20v_list%3A%0A%20%20%20%20%20%20%20%20flat_vowels.append%28v%29%0Aflat_vowels&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 拆解程式執行的每個步驟。

# In[26]:


vowels = [["a", "e", "i", "o", "u"], ["A", "E", "I", "O", "U"]]
flat_vowels = list()
for v_list in vowels:
    for v in v_list:
        flat_vowels.append(v)
flat_vowels


# ## 運用 `dict[key]=value` 合併元素成為 `dict`
# 
# 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=days_of_week%20%3D%20%5B%22Sunday%22,%20%22Monday%22,%20%22Tuesday%22,%20%22Wednesday%22,%20%22Thursday%22,%20%22Friday%22,%20%22Saturday%22%5D%0Adays_of_week_dict%20%3D%20dict%28%29%0Afor%20day%20in%20days_of_week%3A%0A%20%20%20%20day_upper%20%3D%20day.upper%28%29%20%23%20upper-case%0A%20%20%20%20day_abbreviation%20%3D%20day_upper%5B%3A3%5D%20%23%20abbreviation%0A%20%20%20%20days_of_week_dict%5Bday_abbreviation%5D%20%3D%20day%0Adays_of_week_dict&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 拆解程式執行的每個步驟。

# In[27]:


days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
days_of_week_dict = dict()
for day in days_of_week:
    day_upper = day.upper() # upper-case
    day_abbreviation = day_upper[:3] # abbreviation
    days_of_week_dict[day_abbreviation] = day
days_of_week_dict


# ## 以兩個保留字調整迴圈的重複執行次數
# 
# 1. `break` 保留字可以提早結束。
# 2. `continue` 保留字可以略過某些執行次數。

# ## 遇到星期四提早結束
# 
# 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=days_of_week%20%3D%20%5B%22Sunday%22,%20%22Monday%22,%20%22Tuesday%22,%20%22Wednesday%22,%20%22Thursday%22,%20%22Friday%22,%20%22Saturday%22%5D%0Afor%20day%20in%20days_of_week%3A%0A%20%20%20%20if%20day%20%3D%3D%20%22Thursday%22%3A%0A%20%20%20%20%20%20%20%20break%0A%20%20%20%20print%28day%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 拆解程式執行的每個步驟。

# In[28]:


days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
for day in days_of_week:
    if day == "Thursday":
        break
    print(day)


# ## 略過週末
# 
# 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=days_of_week%20%3D%20%5B%22Sunday%22,%20%22Monday%22,%20%22Tuesday%22,%20%22Wednesday%22,%20%22Thursday%22,%20%22Friday%22,%20%22Saturday%22%5D%0Afor%20day%20in%20days_of_week%3A%0A%20%20%20%20if%20day%20in%20%7B%22Sunday%22,%20%22Saturday%22%7D%3A%0A%20%20%20%20%20%20%20%20continue%0A%20%20%20%20print%28day%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 拆解程式執行的每個步驟。

# In[29]:


days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
for day in days_of_week:
    if day in {"Sunday", "Saturday"}:
        continue
    print(day)


# ## 組織程式碼希望達到的目標最主要有兩個：
# 
# 1. 提高程式碼的「可利用性」。
# 2. 減少程式碼的「重複性」。

# ## Python 提供三種機制供使用者組織程式碼
# 
# 視應用範疇由小到大依序為：
# 
# 1. 函數（Function）。
# 2. 類別（Class）。
# 3. 模組（Module）。

# ## 如何理解程式碼組織機制的層次
# 
# - 數行程式碼可以組織為一個函數。
# - 數個函數可以組織為一個類別。
# - 數個函數或類別可以組織為一個模組。
# - 數個模組可以組織為一個功能更多的模組。

# ## 什麼是函數
# 
# 一段被賦予名稱的程式碼，能夠完成某一個文字處理或者數值計算任務，在使用函數之前，必須先確定這個函數在執行的環境中已經被定義妥善。

# ## 函數有四個來源
# 
# 1. 來自內建函數。
# 2. 來自標準模組。
# 3. 來自第三方模組。
# 4. 來自使用者的定義。

# ## 自行定義函數：來自使用者的定義

# In[30]:


def power(x, n):
    out = x**n
    return out

try:
    print(power(5, 3))
except NameError as error_message:
    print(error_message)


# ## 如何自行定義函數
# 
# - `def` 保留字用來定義函數的名稱。
# - 縮排部分稱為程式區塊（Code block），是函數的主體，也是練習題要學員運用預期輸入與參數來完成的部分。
# - 不要忘記把函數的預期輸出寫在 `return` 保留字後。
# - 函數的類別提示（Typing）並不是必要的，但它能幫助學員更快理解練習題。

# ## 自行定義函數的結構
# 
# ```python
# def function_name(INPUTS: TYPE, ARGUMENTS: TYPE) -> TYPE:
#     ### BEGIN SOLUTION
#     OUTPUTS = INPUTS (+-*/...) ARGUMENTS
#     return OUTPUTS
#     ### END SOLUTION
# ```

# ## 定義與使用函數的差別
# 
# - 完成定義函數以後，還需要使用函數才會將引數傳入運算。
# - 定義函數的當下只有與語法錯誤相似的錯誤（例如縮排錯誤）會發起例外。
# - 如果是執行錯誤，在使用函數時才會發起。

# ## 透過 [pythontutor.com](https://pythontutor.com/visualize.html#code=def%20power%28x,%20n%29%3A%0A%20%20%20%20out%20%3D%20x**n%0A%20%20%20%20return%20out%0A%0Apower%285,%203%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false) 觀察

# In[31]:


def power(x, n):
    out = x**n
    return out

power(5, 3)
#power("5", 3)


# ## 彈性參數

# ## 自行定義函數很重要的任務是釐清「預期輸入」與「預期輸出」的對應關係
# 
# 1. 一個輸入 vs. 一個輸出。
# 2. 一個輸入 vs. 多個輸出。
# 3. 多個輸入 vs. 一個輸出。
# 4. 多個輸入 vs. 多個輸出。

# ## 「一個輸入」或者「一個輸出」的對應關係單純且容易理解
# 
# - 多個輸出。
# - 多個輸入。

# ## 以資料結構類別處理函數的「多個輸出」
# 
# - 預設以 `tuple` 資料結構類別應對多個輸出。
# - 可以自行調整偏好的資料結構類別。

# In[32]:


def get_first_and_last_characters(x):
    first_character = x[0]
    last_character = x[-1]
    return first_character, last_character # did not specify a tuple with ()

print(get_first_and_last_characters("Python"))
print(type(get_first_and_last_characters("Python")))


# ## 指定用 `list` 輸出

# In[33]:


def get_first_and_last_characters(x):
    first_character = x[0]
    last_character = x[-1]
    return [first_character, last_character] # specify a list with []

print(get_first_and_last_characters("Python"))
print(type(get_first_and_last_characters("Python")))


# ## 指定用 `dict` 輸出

# In[34]:


def get_first_and_last_characters(x):
    first_character = x[0]
    last_character = x[-1]
    output = {
        "first": first_character,
        "last": last_character
    }
    return output # specify a dict

print(get_first_and_last_characters("Python"))
print(type(get_first_and_last_characters("Python")))


# ## 以資料結構類別或彈性參數處理函數的「多個輸入」
# 
# - 運用資料結構類別作為一個輸入物件名稱。
# - 運用彈性參數。

# ## 運用資料結構類別作為一個輸入物件名稱

# In[35]:


def sum_and_square(x):
    summation = sum(x)
    output = summation**2
    return output

print(sum_and_square([2, 3, 5]))    # [2, 3, 5] as input
print(sum_and_square((2, 3, 5, 7))) # (2, 3, 5, 7) as input


# ## 利用 `*` 標註彈性參數
# 
# - `args` 可以在函數程式區塊中作為一個 `tuple` 供運用。
# - `args` 可以任意使用偏愛的命名。

# In[36]:


def sum_and_square(*args):
    print(type(args))
    summation = sum(args)
    output = pow(summation, 2)
    return output

print(sum_and_square(2, 3, 5))    # 2, 3, 5 as input
print(sum_and_square(2, 3, 5, 7)) # 2, 3, 5, 7 as input


# ## 利用 `**` 標註具有「鍵」與「值」的彈性參數
# 
# - `kwargs` 可以在函數程式區塊中作為一個 `dict` 供運用。
# - `kwargs` 可以任意使用偏愛的命名。

# In[37]:


def print_country_capital(**kwargs):
    print(type(kwargs)) # dict
    for key, value in kwargs.items():
        print(f"Country: {key} Capital: {value}")

print_country_capital(JPN="Tokyo", USA="Washington D.C.", TWN="Taipei")


# ## 作用域

# ## 什麼是作用域
# 
# > 作用域是物件名稱與物件實例參照保持有效的程式碼。
# 
# 來源：<https://en.wikipedia.org/wiki/Scope_(computer_science)>

# ## 當自行定義函數出現在程式中，物件的作用域就會一分為二
# 
# 1. 區域物件（Local objects）。
# 2. 全域物件（Global objects）。

# ## 區域物件僅在附屬於該函數的程式區塊中才有效
# 
# - 函數的輸入與參數物件。
# - 在函數的程式區塊中建立的物件。

# In[38]:


def power(local_x, local_n):
    local_out = local_x**local_n
    print(local_x)   # effective
    print(local_n)   # effective
    print(local_out) # effective

power(-5, 3)


# In[39]:


try:
    #print(local_x)   # non-effective
    #print(local_n)   # non-effective
    print(local_out) # non-effective
except NameError as error_message:
    print(error_message)


# ## 其他函數的區域物件也無效

# In[40]:


def absolute():
    if local_x < 0: # non-effective
        return -local_x
    else:
        return local_x

try:
    absolute()
except NameError as error_message:
    print(error_message)


# ## 不是在函數的程式區塊中建立的是全域物件，在任何地方都有效
# 
# - 不附屬於函數的物件。
# - 定義妥善的函數。

# In[41]:


def power():
    return global_out
def absolute():
    if global_x < 0:
        return -global_x
    else:
        return global_x

global_x = -5
global_n = 3
global_out = global_x**global_n
print(power())
print(absolute())


# ## 物件命名的參照區域物件優先於全域物件
# 
# - 乍看之下在函數的程式區塊中使用全域物件很方便，但這樣的做法並不被推薦。
# - 好的做法是透過函數所設計的參數將全域物件傳入。

# ## 不推薦直接使用全域物件的原因
# 
# - 如果區域以及全域存在相同的物件命名，函數會優先參照區域物件。
# - 避免物件命名的混淆。

# In[42]:


def power(x, n):
    out = x**n
    print(x)
    print(n)
    print(out)

x = 2
n = 4
out = x**n
print(x)     # global x
print(n)     # global n
print(out)   # global out


# In[43]:


power(-5, 3) # local x, n, out


# ## 定義妥善的函數對其他的函數而言就像一個全域的內建函數
# 
# - 可以在自行定義函數的過程中使用其他定義妥善的函數。
# - 讓函數彼此分工，不需要把所有的運算都集中在一個函數中完成。
# - 減少重複的程式碼。
# 
# ```python
# def function_one():
#     ...
#     return ...
#     
# def function_two():
#     ...
#     function_one()
#     ...
#     return ...
# ```

# ## 例如定義「計算中位數」函數時可以使用「取出中位元素」函數

# In[44]:


def retrieve_middle_elements(x):
    length = len(x)
    middle_index = length // 2
    if length % 2 == 1:
        return x[middle_index]
    else:
        return x[middle_index - 1], x[middle_index]
def calculate_median(x):
    middle_elements = retrieve_middle_elements(x) # use retrieve_middle_elements() to get middle elements
    if type(middle_elements) == tuple:
        return sum(middle_elements) / 2
    else:
        return middle_elements

print(calculate_median([9, 8, 3, 6, 7, 3, 1]))
print(calculate_median([1, 3, 2, 5, 4, 9, 8, 6]))


# ## 例如定義「前 100 個 Fizz buzz」函數時可以使用「Fizz buzz」函數

# In[45]:


def fizz_buzz(x):
    if x % 15 == 0:
        return "Fizz Buzz"
    elif x % 3 == 0:
        return "Fizz"
    elif x % 5 == 0:
        return "Buzz"
    else:
        return x
def create_first_100_fizz_buzz():
    fizz_buzz_list = list()
    for integer in range(1, 101):
        fizz_buzz_list.append(fizz_buzz(integer)) # use fizz_buzz() to get fizz buzz element
    return fizz_buzz_list

print(create_first_100_fizz_buzz())


# ## `return` 保留字的兩個作用
# 
# 1. 回傳函數的預期輸出。
# 2. 為函數的程式區塊畫下終止符。

# ## 回傳函數的預期輸出
# 
# - 沒有 `return` 的函數事實上的輸出是 `None`
# - 這也是練習題如果沒有將預期輸出寫在 `return` 保留字後，無法通過批改測試的原因。

# In[46]:


def power(x, n):
    """
    Equivalent to x raised to the power of n.
    """
    output = x**n

type(power(5, 3))


# ## 為函數的程式區塊畫下終止符
# 
# 即便寫在縮排的函數程式區塊之中，`return` 後所寫的程式並沒有作用。

# In[47]:


def power(x, n):
    """
    Equivalent to x raised to the power of n.
    """
    out = x**n
    return out
    print(x)
    print(n)

power(5, 3)


# ## 類別

# ## 什麼是類別
# 
# 自行設計資料或者資料結構的機制，能夠將多個函數與資料組織起來使用，定義「類別」也是入門物件導向程式設計的第一步。

# ## 兩種不同的程式設計
# 
# 1. 程序型程式設計（Procedural programming）。
# 2. 物件導向程式設計（Object-oriented programming, OOP）。

# ## 程序型程式設計
# 
# 以函數為主體的撰寫程式型態稱為「程序型程式設計（Procedural programming）」，把即將要執行的程式碼組織為函數，並依序使用這些函數來完成任務。
# 
# ```python
# def function_one():
#     ...
#     return ...
#     
# def function_two():
#     ...
#     return ...
# 
# function_one()
# function_two()
# ```

# ## 物件導向程式設計
# 
# 除了程序型程式設計，另外一種在軟體開發中被採用的撰寫程式型態稱為「物件導向程式設計（Object-oriented programming, OOP）」。
# 
# ```python
# class class_one:
#     def method_one(self):
#         ...
#         return ...
# 
# object_one = class_one()
# object_one.method_one()
# ```

# ## 程序型程式設計 vs. 物件導向程式設計
# 
# - 線狀 vs. 放射狀。
# - 點餐式 vs. 自助餐式。

# ## 定義類別是一種讓使用者自行設計資料或資料結構的機制。
# 
# - `type()` 內建函數所顯示的 `class` 就是「類別」。
# - 物件（Object）是類別（Class）的實例（Instance），因此建立物件的程式碼常被稱為實例化（Instantiation）。

# ## `luke` 物件是 `str` 類別的實例

# In[48]:


luke = "Luke Skywalker" # instantiation
type(luke)


# ## `skywalkers` 物件是 `list` 類別的實例

# In[49]:


skywalkers = ["Luke Skywalker", "Anakin Skywalker", "Darth Vadar"] # instantiation
type(skywalkers)


# ## 類別之於物件的關係
# 
# - 類別如同藍圖一般的存在。
# - 物件如同依照藍圖所創造的產品。

# ## 為什麼需要定義類別
# 
# - 當內建類別或模組所提供的類別無法滿足需求時，我們定義類別。
# - 定義「類別」是物件導向程式設計的第一步。

# ## 常用的內建類別
# 
# - 資料
#     - `int`
#     - `float`
#     - `str`
#     - `bool`
#     - `NoneType`

# ## 常用的內建類別（續）
# 
# - 資料結構
#     - `list`
#     - `tuple`
#     - `dict`
#     - `set`

# ## 資料科學模組主要提供的類別
# 
# - 資料結構：
#     - `ndarray`
#     - `Index`
#     - `Series`
#     - `DataFrame`

# ## 資料科學模組主要提供的類別（續）
# 
# - 視覺化類別：
#     - `Figure`
#     - `AxesSubplot`
# - 機器學習估計器類別：
#     - 轉換器
#     - 預測器

# ## 設計類別時可以定義函數與資料
# 
# - 在類別程式區塊中定義的函數，實例化後稱為物件的方法（Methods）。
# - 在類別程式區塊中定義的資料，實例化後稱為物件的屬性（Attributes）。
# 
# ```python
# object_name = class_name()
# object_name.method_name()
# object_name.attribute_name
# ```

# ## 使用內建函數 `dir()` 檢視物件的方法與屬性
# 
# - 前後有兩個底線 `__` 命名的方法或屬性是所謂的特殊方法、特殊屬性。
# - 特殊方法或屬性具有 Python 指定好的功能。

# In[50]:


# object luke is an instance of str class
luke = "Luke Skywalker"
print(dir(luke))


# In[51]:


# object skywalkers is an instance of list class
skywalkers = ["Luke Skywalker", "Anakin Skywalker", "Darth Vadar"]
print(dir(skywalkers))


# ## 模組

# ## 複習：Python 禪學（The Zen of Python）
# 
# - `import` 是 Python 的保留字（Keywords），可以載入模組。
# - `this` 是 Python 的一個標準模組，可以印出 Python 禪學。

# In[52]:


import this


# ## 什麼是模組
# 
# 模組（Module）指的是以檔案或資料夾形式，來組織 Python 的函數以及類別。檔案形式可以對應「數個函數或類別可以組織為一個模組」層次、資料夾形式可以對應「數個模組可以組織為一個功能更多的模組」層次。

# ## 模組有三個來源
# 
# 1. 標準模組。
# 2. 來自使用者的定義。
# 3. 第三方模組。

# ## 標準模組（Standard libraries）
# 
# - 伴隨 Python 直譯器（來自 [python.org](https://www.python.org) 的版本）安裝的模組。
# - 可以直接載入並使用。
# - 有哪些標準模組可以直接載入並使用：<https://docs.python.org/3/library>

# ## 使用 `import`、`from` 與 `as` 保留字
# 
# - 使用 `import` 載入模組。
# - 使用 `from module import function/class` 載入模組中特定的函數或類別。
# - 使用 `as` 調整模組、函數或者類別的命名。

# ## 使用 `import` 載入模組
# 
# - 以 `module` 命名使用。
# - 使用模組中的函數、類別時都以 `module.function()` 或 `module.class()` 來參照命名。
# 
# ```python
# import module
# 
# module.function()
# object = module.class()
# ```

# In[53]:


import os

os.getcwd()


# In[54]:


import datetime

first_day_of_2026 = datetime.date(2026, 1, 1)
print(first_day_of_2026)


# ## 使用 `from module import function/class` 載入模組中特定的函數或類別
# 
# 載入特定函數或類別，就直接以 `function()` 或 `class()` 來參照命名。
# 
# ```python
# from module import function
# from module import class
# 
# function()
# object = class()
# ```

# In[55]:


from os import getcwd

getcwd()


# In[56]:


from datetime import date

first_day_of_2026 = date(2026, 1, 1)
print(first_day_of_2026)


# ## 使用 `as` 調整模組、函數或者類別的命名
# 
# ```python
# import module as module_alias
# from module import function as function_alias
# from module import class as class_alias
# 
# module_alias.function()
# function_alias()
# object = class_alias()
# ```

# In[57]:


import os as operating_system
from os import getcwd as get_current_working_directory

print(operating_system.getcwd())
print(get_current_working_directory())


# In[58]:


import datetime as dttm
from datetime import date as dt

first_day_of_2026 = dttm.date(2026, 1, 1)
print(first_day_of_2026)
first_day_of_2026 = dt(2026, 1, 1)
print(first_day_of_2026)


# ## 如何決定模組載入的形式
# 
# - `import module`
# - `import module as alias`
# - `from module import function/class`
# - `from module import function/class as alias`

# ## 如何決定模組載入的形式（續）
# 
# - 依照模組說明文件（Documentation）中的範例決定。
# - 例如：使用 `os` 模組時採用 `import os`
# - 例如：使用 `datetime` 模組時採用 `from datetime import function/class`
# 
# 來源：<https://docs.python.org/3/library/os.html#file-object-creation>, <https://docs.python.org/3/library/datetime.html#examples-of-usage-date>

# In[59]:


import os

os.getcwd()


# In[60]:


from datetime import date

first_day_of_2026 = date(2026, 1, 1)
print(first_day_of_2026)


# ## 為什麼模組名稱可以被認得
# 
# - Python 會搜尋標準模組的安裝路徑中是否有檔名為 `datetime.py`、`os.py`、`this.py` 或資料夾名稱 `this`、`os`、`datetime`（事實上這裡舉例的模組形式均為檔案名稱）。
# - 如果沒有，就會產生找不到模組錯誤（`ModuleNotFoundError`）。
# - 如果有，就會依據指令載入裡面全部或特定的函數與類別。

# In[61]:


# There is a this.py, but no that.py
try:
    import that
except ModuleNotFoundError as error_message:
    print(error_message)


# ## 隨堂練習

# ## 隨堂練習
# 
# <https://colab.research.google.com/github/datainpoint/classroom-fintech-bot-2026/blob/main/03-exercises.ipynb>
