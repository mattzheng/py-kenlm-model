# py-kenlm-model
python | 高效使用统计语言模型kenlm：新词发现、分词、智能纠错等

之前看到苏神[【重新写了之前的新词发现算法：更快更好的新词发现】](https://spaces.ac.cn/archives/6920)中提到了kenlm，之前也自己玩过，没在意，现在遇到一些大规模的文本问题，模块确实好用，前几天还遇到几个差点“弃疗”的坑，解决了之后，就想，不把kenlm搞明白，对不起我浪费的两天。。

**kenlm的优点（[关于kenlm工具训练统计语言模型](https://blog.csdn.net/HHTNAN/article/details/84231733)）：**
训练语言模型用的是传统的“统计+平滑”的方法，使用kenlm这个工具来训练。它快速，节省内存，最重要的是，允许在开源许可下使用多核处理器。
kenlm是一个C++编写的语言模型工具，具有速度快、占用内存小的特点，也提供了Python接口。

额外需要加载的库：
```
kenlm
pypinyin
pycorrector
```

笔者的代码可见github，只是粗略整理，欢迎大家一起改:
[mattzheng/py-kenlm-model](https://github.com/mattzheng/py-kenlm-model)

相关新词发现,fork了苏神的,进行了微调:

[mattzheng/word-discovery](https://github.com/mattzheng/word-discovery)

博客链接：

[python | 高效使用统计语言模型kenlm：新词发现、分词、智能纠错等](https://mattzheng.blog.csdn.net/article/details/101512616)



----------



# 1 kenlm安装

在这里面编译：[kpu/kenlm](https://github.com/kpu/kenlm)，下载库之后编译：
```python
mkdir -p build
cd build
cmake ..
make -j 4
```
一般编译完，很多有用的文件都存在`build/bin`之中，这个后面会用到：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190927094924188.png)
python库的安装方式：

```python
pip install https://github.com/kpu/kenlm/archive/master.zip
```
简单使用：

```python
import kenlm
model = kenlm.Model('lm/test.arpa')
print(model.score('this is a sentence .', bos = True, eos = True))
```

坑点来了，笔者之前装在docker之中的，之前一不小心重启docker，kenlm就不灵了。。
当时并不知道该如何重新编译，就重新：`cmake ..` + `make -j 4`，但是这样出来，运行会报很多依赖没装：

```python

libboost_program_options.so.1.54.0: cannot open shared object file: No such file or directory
```
笔者还假了嘛嘎的去ubuntu上拉下来装了，又报其他依赖错。。

（此处省略N多次，无效尝试。。。）

如果出现：

```python
-- Could NOT find BZip2 (missing:  BZIP2_LIBRARIES BZIP2_INCLUDE_DIR) 
-- Could NOT find LibLZMA (missing:  LIBLZMA_INCLUDE_DIR LIBLZMA_LIBRARY LIBLZMA_HAS_AUTO_DECODER LIBLZMA_HAS_EASY_ENCODER LIBLZMA_HAS_LZMA_PRESET)

```
需安装：

```python
sudo apt install libbz2-dev
sudo apt install liblzma-dev
```

之后实验发现，把`build`文件夹删了，重新来一遍`cmake ..` + `make -j 4`即可。


----------

# 2 kenlm统计语言模型使用

## 2.1 kenlm的训练 `lmplz`
### 2.1.1 两种训练方式
训练是根据`build/bin/lmplz `来进行，一般来说有两种方式：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190927143540315.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9tYXR0emhlbmcuYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)

（1）管道的方式传递

数据print的方式，苏神之前的博客【[【中文分词系列】 5. 基于语言模型的无监督分词](https://spaces.ac.cn/archives/3956#%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83)】中有提到：

```python
python p.py|./kenlm/bin/lmplz -o 4 > weixin.arpa
```
p.py为：

```python
import pymongo
db = pymongo.MongoClient().weixin.text_articles

for text in db.find(no_cursor_timeout=True).limit(500000):
    print ' '.join(text['text']).encode('utf-8')
```

（2）预先生成语料文本

直接命令行，数据保存
```python
bin/lmplz -o 3 --verbose_header --text ../text-18-03/text_18-03-AU.txt --arpa MyModel/log.arpa
```
其中参数的大致意义：

```python
-o n:最高采用n-gram语法
-verbose_header:在生成的文件头位置加上统计信息
--text text_file:指定存放预料的txt文件
--arpa:指定输出的arpa文件
-S [ --memory ] arg (=80%)  Sorting memory内存预占用量
--skip_symbols : Treat <s>, </s>, and <unk> as whitespace instead of throwing an  exception
```

预先语料可以不加开头、结尾符号，其中， 需要特别介绍三个特殊字符。
`<s>、</s>和<unk>`
`<s>`和`</s>`结对使用，模型在计算概率时对每句话都进行了处理，将该对标记加在一句话的起始和结尾。
这样就把开头和结尾的位置信息也考虑进来。
如`“我 喜欢 吃 苹果” --> "<s> 我 喜欢 吃 苹果 </s>"`
`<unk>`表示unknown的词语，对于oov的单词可以用它的值进行替换。

可参考：
不带开头结尾：
```
W h o o   后   拱 辰 享 水   水 妍 护 肤 套 装 整 套 质 地 都 比 较 清 爽 
 滋 润 
 侧 重 保 湿 
 适 合 各 种 肤 质 
 调 节 肌 肤 水 平 衡 
 它 还 具 有 修 复 功 效 
 提 亮 肤 色 我 是 油 性 肤 质 用 起 来 也 一 点 也 不 觉 得 油 腻 
 味 道 淡 淡 的 还 很 好 闻 
 也 很 好 吸 收 
 质 地 清 爽 
```
带开头结尾的：

```python
<s> 3 乙 方 应 依 据 有 关 法 律 规 定 </s>
<s> 对 甲 方 为 订 立 和 履 行 本 合 同 向 乙 方 提 供 的 有 关 非 公 开 信 息 保 密 </s>
<s> 但 下 列 情 形 除 外 </s>
<s> 1 贷 款 人 有 权 依 据 有 关 法 律 法 规 或 其 他 规 范 性 文 件 的 规 定 或 金 融 监 管 机 构 的 要 求 </s>
```

具体的训练过程可见该博客：[图解N-gram语言模型的原理--以kenlm为例](https://blog.csdn.net/asrgreek/article/details/81979194)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190927143223581.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9tYXR0emhlbmcuYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)



### 2.1.2 生成文件arpa的解释
来源：[语言模型kenlm的训练及使用](https://www.bbsmax.com/A/WpdKmENJVQ/)
其中生成的arpa文件有：

```python


    \1-grams:
    -6.5514092	<unk>	0
    0	<s>	-2.9842114
    -1.8586434	</s>	0
    -2.88382	!	-2.38764
    -2.94351	world	-0.514311
    -2.94351	hello	-0.514311
    -6.09691	guys	-0.15553
     
    \2-grams:
    -3.91009	world !	-0.351469
    -3.91257	hello world	-0.24
    -3.87582	hello guys	-0.0312
     
    \3-grams:
    -0.00108858	hello world !
    -0.000271867	, hi hello !
     
    \end\


```

介绍该文件需要引入一个新的概念，back_pro【[language model](http://blog.csdn.net/visionfans/article/details/50131397)】
三个字段分别是：`Pro,word,back_pro `
注：arpa文件中给出的数值都是以10为底取对数后的结果




### 2.1.3 几个训练坑点解读 

划重点来了，其中`-s` 非常重要，默认是`80%`，如果机器有20%被占了，笔者当时发现，10句话训练模型也能超内存，这不是瞎胡闹：

```python
#34304 what():  /mnt/mNLP/kg/kenlm/util/scoped.cc:20 in void* util::{anonymous}::InspectAddr(void*, std::size_t, const char*) threw MallocException because `!addr && requested'.
#Cannot allocate memory for 84881776616 bytes in malloc
```
需要额外设置内存占用量！当然还有挺多可能会产生意外的参数：

参数     | 解释
-------- | :-----
minimum_block arg (=8K)  | Minimum block size to allow
sort_block arg (=64M)  | Size of IO operations for sort  (determines arity)
block_count arg (=2)  | Block count (per order)
interpolate_unigrams [=arg(=1)] (=1) | Interpolate the unigrams (default) as  opposed to giving lots of mass to <unk>  like SRI.  If you want SRI's behavior with a large <unk> and the old lmplz  default, use --interpolate_unigrams 0.
discount_fallback [=arg(=0.5 1 1.5)] | The closed-form estimate for Kneser-Ney  discounts does not work without  singletons or doubletons. 
。。。(还有不少) | 。。。

还有可能会报错：

```python
Unigram tokens 153 types 116
    === 2/5 Calculating and sorting adjusted counts ===
    Chain sizes: 1:1392 2:10964970496 3:20559319040 4:32894910464
    /mnt/mNLP/kg/kenlm/lm/builder/adjust_counts.cc:52 in void lm::builder::{anonymous}::StatCollector::CalculateDiscounts(const lm::builder::DiscountConfig&) threw BadDiscountException because `s.n[j] == 0'.
    Could not calculate Kneser-Ney discounts for 1-grams with adjusted count 4 because we didn't observe any 1-grams with adjusted count 3; Is this small or artificial data?
    Try deduplicating the input.  To override this error for e.g. a class-based model, rerun with --discount_fallback
```
报错码为:34304,主要是因为字数太少，所以训练的时候需要多加一些。

## 2.2 模型压缩二进制化`build_binary `
这边生成的arpa文件，可能会比较大，可以通过二进制化缩小文件大小：

```python
bin/build_binary -s lm.arpa lm.bin
```
将arpa文件转换为binary文件，这样可以对arpa文件进行压缩，提高后续在python中加载的速度。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190927143614915.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9tYXR0emhlbmcuYmxvZy5jc2RuLm5ldA==,size_16,color_FFFFFF,t_70)
虽然大小没有发生太大的变化，但是压缩后会大大提高Python加载的速度。

可能会报错，报错码为：256，原因如下：

```python
No such file or directory while opening output/test2.arpa
```



## 2.3 利用kenlm的`count_ngrams`计算n-grams
苏神[【重新写了之前的新词发现算法：更快更好的新词发现】](https://spaces.ac.cn/archives/6920)中用的是这个。
这个库存在`build/bin/count_ngrams`
```python
    # Counts n-grams from standard input.
    # corpus count:
    #   -h [ --help ]                     Show this help message
    #   -o [ --order ] arg                Order
    #   -T [ --temp_prefix ] arg (=/tmp/) Temporary file prefix
    #   -S [ --memory ] arg (=80%)        RAM
    #   --read_vocab_table arg            Vocabulary hash table to read.  This should
    #                                     be a probing hash table with size at the 
    #                                     beginning.
    #   --write_vocab_list arg            Vocabulary list to write as null-delimited 
    #                                     strings.
```
其中也有该死的`-s`，要留意。
执行命令示例：
```python
./count_ngrams -S 50% -o 4 --write_vocab_list output/test2.chars <output/test2.corpus >output/test2.ngrams
```
其中，参数`-s`,`-o`与前面一样，
输入的是预生成文本`output/test2.corpus`，生成两个文件：`output/test2.chars` 和 `output/test2.ngrams`，分别是单词文件和ngrams的文件集合

其中，执行的时候，如果返回的不是0，都是有错误的，笔者自己遇到过几个：

错误码     | 原因
-------- | :-----
32256| 计算不了 - 错误类型：没有权限，可能是count_ngrams没有执行权限
32512| 依赖报错：/count_ngrams: error while loading shared libraries: libboost_program_options.so.1.58.0: cannot open shared object file: No such file or directory
34304  | 内存报错：Cannot allocate memory for 84881776616 bytes in malloc


----------


# 3 kenlm模型的初级使用

参考文档：[kenlm/python/example.py](https://github.com/kpu/kenlm/blob/master/python/example.py)

## 3.1 model.score函数
python已经有可以使用的库，安装教程见第1章，简单测试方式：

```python
import kenlm
model = kenlm.Model('lm/test.arpa')
print(model.score('this is a sentence .', bos = True, eos = True))
```
其中，
每个句子通过语言模型都会得到一个概率(0-1),然后对概率值取log得到分数(-\propto ,0],得分值越接近于0越好。
score函数输出的是对数概率，即log10(p('微 信'))，其中字符串可以是gbk，也可以是utf-8
`bos=False, eos=False`意思是不自动添加句首和句末标记符,得分值越接近于0越好。
一般都要对计算的概率值做log变换，不然连乘值太小了，在程序里会出现 inf 值。

> @param sentence is a string (do not use boundary symbols)
@param bos should kenlm add a bos state
@param eos should kenlm add an eos state
来源：https://github.com/kpu/kenlm/blob/master/python/kenlm.pyx

该模块，可以用来测试词条与句子的通顺度：

```python
text = '再 心 每 天也 不 会 担 个 大 油 饼 到 了 下  午 顶 着 一 了 '
model.score(text, bos=True, eos=True)
```
需要注意，是需要空格隔开的。

## 3.2 model.full_scores函数
`score`是`full_scores`是精简版，full_scores会返回：` (prob, ngram length, oov)`
包括：概率，ngram长度，是否为oov

```
# Show scores and n-gram matches
sentence = '盘点不怕被税的海淘网站❗️海淘向来便宜又保真，比旗舰店、专柜和代购好太多！'

words = ['<s>'] + parse_text(sentence).split() + ['</s>']
for i, (prob, length, oov) in enumerate(model.full_scores(sentence)):
    print('{0} {1}: {2}'.format(prob, length, ' '.join(words[i+2-length:i+2])))
    if oov:
        print('\t"{0}" is an OOV'.format(words[i+1]))

# Find out-of-vocabulary words
for w in words:
    if not w in model:
        print('"{0}" is an OOV'.format(w))
```

## 3.3 kenlm.State()状态转移概率

```python
'''
状态的累加
score defaults to bos = True and eos = True.  
Here we'll check without the endof sentence marker.  
'''
#Stateful query
state = kenlm.State()
state2 = kenlm.State()
#Use <s> as context.  If you don't want <s>, use model.NullContextWrite(state).
model.BeginSentenceWrite(state)
```
然后还有：

```python
accum = 0.0
accum += model.BaseScore(state, "海", state2)
print(accum)
accum += model.BaseScore(state2, "淘", state)
print(accum)
accum += model.BaseScore(state, "</s>", state2)
print(accum)

>>>-3.0864107608795166
>>>-3.6341209411621094
>>>-4.645392656326294

model.score("海 淘", eos = False)
>>> -3.381103515625
```
这个实验可以看到：state2的状态概率与score的概率差不多，该模块还有很多可以深挖，NSP任务等等。


## 3.4 语句通顺度检测
通顺度其实用score即可，只不过用整个句子，整个句子需要空格隔开。
这边有一个项目，还封装了API，可参考：[DRUNK2013/lm-ken](https://github.com/DRUNK2013/lm-ken)



----------


# 4 kenlm的深度使用 - 分词
参考于：[【中文分词系列】 5. 基于语言模型的无监督分词](https://spaces.ac.cn/archives/3956#%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83)
苏神的代码模块：

```python
import kenlm
model = kenlm.Model('weixin.klm')

from math import log10

#这里的转移概率是人工总结的，总的来说，就是要降低长词的可能性。
trans = {'bb':1, 'bc':0.15, 'cb':1, 'cd':0.01, 'db':1, 'de':0.01, 'eb':1, 'ee':0.001}
trans = {i:log10(j) for i,j in trans.iteritems()}

def viterbi(nodes):
    paths = nodes[0]
    for l in range(1, len(nodes)):
        paths_ = paths
        paths = {}
        for i in nodes[l]:
            nows = {}
            for j in paths_:
                if j[-1]+i in trans:
                    nows[j+i]= paths_[j]+nodes[l][i]+trans[j[-1]+i]
            k = nows.values().index(max(nows.values()))
            paths[nows.keys()[k]] = nows.values()[k]
    return paths.keys()[paths.values().index(max(paths.values()))]

def cp(s):
    return (model.score(' '.join(s), bos=False, eos=False) - model.score(' '.join(s[:-1]), bos=False, eos=False)) or -100.0

def mycut(s):
    nodes = [{'b':cp(s[i]), 'c':cp(s[i-1:i+1]), 'd':cp(s[i-2:i+1]), 'e':cp(s[i-3:i+1])} for i in range(len(s))]
    tags = viterbi(nodes)
    words = [s[0]]
    for i in range(1, len(s)):
        if tags[i] == 'b':
            words.append(s[i])
        else:
            words[-1] += s[i]
    return words
```
将分词转化为了标注问题，如果字语言模型取到4-gram，那么它相当于做了如下的字标注：

```python
b：单字词或者多字词的首字
c：多字词的第二字
d：多字词的第三字
e：多字词的其余部分
```
笔者基本没改动，微调至py3可用，笔者的模块可以使用的方式为：
```
# 初始化
km = kenlm_model()
km.model = km.load_model('output/test2.klm')
```
查询与分词：
```
sentence = '这瓶洗棉奶用着狠不错'
km.mycut(sentence)
```
当然，分词模块只是for fun的。。


----------

# 5 kenlm的深度使用 - 新词发现

苏神[【重新写了之前的新词发现算法：更快更好的新词发现】](https://spaces.ac.cn/archives/6920)中用的是这个。大部分与苏神一致，微调至py3已经加入分词方式的调用。这个可能需要先训练：

## 5.1 训练语料
**第一步：模型加载**

```python
km = kenlm_model(save_path = 'output',project = 'test2',\
                 memory = '50%',min_count = 2,order = 4,\
                 skip_symbols = '"<unk>"',kenlm_model_path = './kenlm/build/bin/')
```
其中，
- save_path， 是相关文件存储在哪，因为一次性会生成很多临时文件
- project ，是项目编号，编译项目管理
- memory，调用时候占用的内存容量
- min_count = 2，在筛选n-grams最小的频率
- order = 4，n-grams中的n
- skip_symbols = `'"<unk>"'`，Treat` <s>`, `</s>`, and `<unk>` as whitespace instead of throwing an exception
- kenlm_model_path = './kenlm/build/bin/'，kenlm那些编译好的文件存放在的位置

**第二步：准备训练材料**
训练，笔者拿了五句话来做训练（**实际需要多准备一些，不然文字太少会报错**）：

```python
text_list = ['Whoo 后 拱辰享水 水妍护肤套装整套质地都比较清爽，滋润，侧重保湿，适合各种肤质',
 '盘点不怕被税的海淘网站❗️海淘向来便宜又保真，比旗舰店、专柜和代购好太多！还能体验海淘乐趣~外网需要双币信用卡，往往需要转运，北上地区容易被税。',
 '学生用什么洗面奶好？学生党必备的这六款性价比最高的洗面奶是什么？',
 '国货大宝。启初。……使用分享修复：玉泽或至本（第三代玉泽身体乳没有了麦冬根和神经酰胺）。芦荟胶（含酒精，不锁水，偶尔敷一下，皮肤会越用越干）。Swisse蜂蜜面膜（清洁鼻子，效果肉眼可见，不能常用）。',
 '资生堂悦薇乳液，会回购。夏天用略油腻，冬天用刚好。真的有紧致感，28岁，眼部有笑纹，其他地方还可以。这是第二个空瓶。冬天会回购。没有美白效果。(资生堂悦薇)']

km.write_corpus(km.text_generator(text_list,jieba_cut = False), km.corpus_file) # 将语料转存为文本
>>> success writed
```

将文本解析为：
```
W h o o   后   拱 辰 享 水   水 妍 护 肤 套 装 整 套 质 地 都 比 较 清 爽 
 滋 润 
 侧 重 保 湿 
 适 合 各 种 肤 质
```
并保存在：`km.corpus_file`文件之中

**第三步：计算模型的n-grams**
```python
# 计算模型的n-grams
km.count_ngrams() # 用Kenlm统计ngram

>>>success,code is : 0 , 
 code is : ./kenlm/build/bin/build_binary -S 50% -s output/test2.arpa output/test2.klm 

```
这里如果状态码不是0，就是报错了，写在py之中不好看到报错信息，笔者自己把相关执行代码也显示出来，所以自己去终端敲一下，定位问题。这步骤，根据`.corpus`文件，生成`.chars`和`.ngrams`


## 5.2 读入模型并使用
这个`read_ngrams` 和 `filter_ngrams`都是苏神中的代码了
```python
ngrams,total = km.read_ngrams()
ngrams_2 = km.filter_ngrams(ngrams, total, min_pmi=[0, 1, 3, 5])
ngrams_2
```
read_ngrams是读入之前的训练文件，ngrams是有三个grams（1-gram,2-gram,3-gram）的（word freq）词与词频，
filter_ngrams就是过滤ngram了，[0, 2, 4, 6]是互信息的阈值，其中第一个0无意义，仅填充用，而2, 4, 6分别是2gram、3gram、4gram的互信息阈值，基本上单调递增比较好。
得到这些n-grams之后的逻辑与苏神有点不一样，我的逻辑是：
```
是否能够被jieba分开
且限定在一定的条件下：词性限定 + 个别停用字
```
那么使用方式：
```
km.word_discovery(ngrams_2)

>{'缓痘痘': 2,
 '奶参考': 2,
 '中文界': 5,
 '文界面': 5,
 '界面支': 5,
 '蜂蜜面': 2,
 '20英': 2,
 '面奶参': 2,
 '舒缓痘': 2,
 '0英': 4}
```
我这边是返回了词 + 词频，便于画词云。

----------

# 6 kenlm的深度使用 - 智能纠错
部分来源：
[pycorrector](https://github.com/shibing624/pycorrector)
[中文文本纠错算法--错别字纠正的二三事](https://zhuanlan.zhihu.com/p/40806718)

笔者最近在研究智能写作，对纠错还是蛮有需求的，这边有看到文章些kenlm用在纠错上，不过是a/an的简单区别，这边笔者也基于此简单使用了一些。
纠错任务一般分别两个：

- 发现错误
- 改正错误

这边智能纠错笔者比较推荐的库是：[pycorrector](https://github.com/shibing624/pycorrector)，优点很多：

- 一直在维护
- 可自定义加载自己的一些规则
- 有深度方案的选项

当然这个库好像要预装tensorflow？ 需要安装尝试的小伙伴注意下。中文文本纠错任务，常见错误类型包括：

- 谐音字词，如 配副眼睛-配副眼镜
- 混淆音字词，如 流浪织女-牛郎织女
- 字词顺序颠倒，如 伍迪艾伦-艾伦伍迪
- 字词补全，如 爱有天意-假如爱有天意
- 形似字错误，如 高梁-高粱
- 中文拼音全拼，如 xingfu-幸福
- 中文拼音缩写，如 sz-深圳
- 语法错误，如 想象难以-难以想象

因为只是实验，所以，发现错误这个环节就交给pycorrector了，笔者用kenlm来改正错误。
简单的发现错误的环节，思路大概是：
> 错误检测部分先通过结巴中文分词器切词，由于句子中含有错别字，所以切词结果往往会有切分错误的情况，这样从字粒度和词粒度两方面检测错误，整合这两种粒度的疑似错误结果，形成疑似错误位置候选集

Kenlm改正错误，有个好处就是kenlm可以定制化训练某一领域的大规模语料的语言模型。本次简单实验的改正逻辑是：
```
两个字至少有一个字，字形相似
两个字拼音首字母一致
```
所以只是上述提到错误中的拼音缩写修正。

## 6.1 pypinyin拼音模块

其中，拼音模块涉及到了`pypinyin`，用来识别汉字的拼音，还有非常多种的模式：
```
from pypinyin import lazy_pinyin, Style
	# Python 中拼音库 PyPinyin 的用法
	# https://blog.csdn.net/devcloud/article/details/95066038

tts = ['BOPOMOFO', 'BOPOMOFO_FIRST', 'CYRILLIC', 'CYRILLIC_FIRST', 'FINALS', 'FINALS_TONE',
 'FINALS_TONE2', 'FINALS_TONE3', 'FIRST_LETTER', 'INITIALS', 'NORMAL', 'TONE', 'TONE2', 'TONE3']
for tt in tts:
    print(tt,lazy_pinyin('聪明的小兔子吃', style=eval('Style.{}'.format(tt))   ))


```

其中结果为：

```python
BOPOMOFO ['ㄘㄨㄥ', 'ㄇㄧㄥˊ', 'ㄉㄜ˙', 'ㄒㄧㄠˇ', 'ㄊㄨˋ', 'ㄗ˙', 'ㄔ']
BOPOMOFO_FIRST ['ㄘ', 'ㄇ', 'ㄉ', 'ㄒ', 'ㄊ', 'ㄗ', 'ㄔ']
CYRILLIC ['цун1', 'мин2', 'дэ', 'сяо3', 'ту4', 'цзы', 'чи1']
CYRILLIC_FIRST ['ц', 'м', 'д', 'с', 'т', 'ц', 'ч']
FINALS ['ong', 'ing', 'e', 'iao', 'u', 'i', 'i']
FINALS_TONE ['ōng', 'íng', 'e', 'iǎo', 'ù', 'i', 'ī']
FINALS_TONE2 ['o1ng', 'i2ng', 'e', 'ia3o', 'u4', 'i', 'i1']
FINALS_TONE3 ['ong1', 'ing2', 'e', 'iao3', 'u4', 'i', 'i1']
FIRST_LETTER ['c', 'm', 'd', 'x', 't', 'z', 'c']
INITIALS ['c', 'm', 'd', 'x', 't', 'z', 'ch']
NORMAL ['cong', 'ming', 'de', 'xiao', 'tu', 'zi', 'chi']
TONE ['cōng', 'míng', 'de', 'xiǎo', 'tù', 'zi', 'chī']
TONE2 ['co1ng', 'mi2ng', 'de', 'xia3o', 'tu4', 'zi', 'chi1']
TONE3 ['cong1', 'ming2', 'de', 'xiao3', 'tu4', 'zi', 'chi1']
```

可以看出不同的style可以得到不同拼音形式。

## 6.2 pycorrector纠错模块

pycorrector的`detect`，可以返回，错误字的信息
```
import pycorrector
sentence = '这瓶洗棉奶用着狠不错'
idx_errors = pycorrector.detect(sentence)
>>> [['这瓶', 0, 2, 'word'], ['棉奶', 3, 5, 'word']]
```
correct是专门用来纠正：

```python
pycorrector.correct(sentence)
```


## 6.3 pycorrector与kenlm纠错对比

来对比一下pycorrector自带的纠错和本次实验的纠错：

```python
import pycorrector
sentence = '这瓶洗棉奶用着狠不错'
idx_errors = pycorrector.detect(sentence)

correct = []
for ide in idx_errors:
    right_word = km.find_best_word(ide[0],ngrams_,freqs = 0)
    if right_word != ide[0]:
        correct.append([right_word] + ide)

print('错误：',idx_errors)
print('pycorrector的结果：',pycorrector.correct(sentence))
print('kenlm的结果：',correct)

> 错误： [['这瓶', 0, 2, 'word'], ['棉奶', 3, 5, 'word']]
> pycorrector的结果： ('这瓶洗面奶用着狠不错', [['棉奶', '面奶', 3, 5]])
> kenlm的结果： [['面奶', '棉奶', 3, 5, 'word']]
```

其他类似的案例：

```python
sentence =  '少先队员因该给老人让坐'

> 错误： [['因该', 4, 6, 'word'], ['坐', 10, 11, 'char']]
> pycorrector的结果： ('少先队员应该给老人让座', [['因该', '应该', 4, 6], ['坐', '座', 10, 11]])
> kenlm的结果： [['应该', '因该', 4, 6, 'word']]
```

这里笔者的简陋规则暴露问题了，只能对2个字以上的进行判定。

另一个：

```python
sentence = '绿茶净华可以舒缓痘痘机肤'

> 错误： [['净华', 2, 4, 'word'], ['机肤', 10, 12, 'word']]
> pycorrector的结果： ('绿茶净化可以舒缓痘痘肌肤', [['净华', '净化', 2, 4], ['机肤', '肌肤', 10, 12]])
> kenlm的结果： [['精华', '净华', 2, 4, 'word'], ['肌肤', '机肤', 10, 12, 'word']]
```
因为训练的是这方面的语料，要比prcorrector好一些。


----------


# 参考文献

[1 使用kenLM训练语言模型](https://blog.csdn.net/Nicholas_Wong/article/details/80013547)

[2 使用kenlm模型判别a/an错别字](https://zhuanlan.zhihu.com/p/39722203)

[3 语言模型kenlm的训练及使用](https://www.bbsmax.com/A/WpdKmENJVQ/)

[4 DRUNK2013/lm-ken](https://github.com/DRUNK2013/lm-ken)

[5 重新写了之前的新词发现算法：更快更好的新词发现](https://spaces.ac.cn/archives/6920)

[6 【中文分词系列】 5. 基于语言模型的无监督分词](https://spaces.ac.cn/archives/3956#%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83)

[7 自然语言处理 | (13)kenLM统计语言模型构建与应用](https://blog.csdn.net/sdu_hao/article/details/87101741)



