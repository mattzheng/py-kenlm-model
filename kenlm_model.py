import jieba
from collections import Counter
from tqdm import tqdm
import re,glob,os
from math import log10
import struct
import math
import kenlm
from collections import Counter
import jieba.posseg as pseg

class kenlm_model():
    def __init__(self,save_path = 'output',project = 'test2',\
                 memory = '50%',min_count = 5,order = 4,\
                 skip_symbols = '"<unk>"',kenlm_model_path = './kenlm/build/bin/'):
        self.memory = memory    # 运行预占用内存
        self.min_count = min_count # n-grams考虑的最低频率
        self.order = order    # n-grams的数量
        self.kenlm_model_path = kenlm_model_path 
        # kenlm模型路径 / 包括：count_ngrams/lmplz等kenlm模块的路径
        self.corpus_file = save_path + '/%s.corpus'%project # 语料保存的文件名
        self.vocab_file = save_path + '/%s.chars'%project # 字符集保存的文件名
        self.ngram_file = save_path + '/%s.ngrams'%project # ngram集保存的文件名
        self.output_file = save_path + '/%s.vocab'%project # 最后导出的词表文件名
        self.arpa_file = save_path + '/%s.arpa'%project # 语言模型的文件名arpa
        self.klm_file = save_path + '/%s.klm'%project# 语言模型的二进制文件名klm,也可以.bin
        self.skip_symbols = '"<unk>"'   
        # lm_train训练时候，Treat <s>, </s>, and <unk> as whitespace instead of throwing an exception
        #这里的转移概率是人工总结的，总的来说，就是要降低长词的可能性。
        trans = {'bb':1, 'bc':0.15, 'cb':1, 'cd':0.01, 'db':1, 'de':0.01, 'eb':1, 'ee':0.001}
        self.trans = {i:log10(j) for i,j in trans.items()}
        self.model = None

        
    def load_model(self,model_path ):
        return kenlm.Model(model_path)
    
    
    # 语料生成器，并且初步预处理语料
    @staticmethod
    def text_generator(texts,jieba_cut = False ):
        '''
        输入:
            文本,list
        输出:
            ['你\n', '是\n', '谁\n']
        其中:
            参数jieba_cut,代表是否基于jieba分词来判定
        
        '''
        for text in texts:
            text = re.sub(u'[^\u4e00-\u9fa50-9a-zA-Z ]+', '\n', text)
            if jieba_cut:
                yield ' '.join(list(jieba.cut(text))) + '\n'
            else:
                yield ' '.join(text) + '\n'
    
    @staticmethod
    def write_corpus(texts, filename):
        """将语料写到文件中，词与词(字与字)之间用空格隔开
        """
        with open(filename, 'w') as f:
            for s in texts:
                #s = ' '.join(s) + '\n'
                f.write(s)
        print('success writed')
    
    
    def count_ngrams(self):
        """
        通过os.system调用Kenlm的count_ngrams来统计频数
        
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
        """
        #corpus_file,vocab_file,ngram_file,memory = '50%',order = 4
        executive_code = self.kenlm_model_path + 'count_ngrams -S %s -o %s --write_vocab_list %s <%s >%s'%(self.memory,self.order, self.vocab_file, self.corpus_file, self.ngram_file)
        status = os.system(executive_code)
        if status == 0:
            return 'success,code is : %s , \n code is : %s '%(status,executive_code)
        else:
            return 'fail,code is : %s ,\n code is : %s '%(status,executive_code)
    
    
    def lm_train(self):
        '''
        # 训练数据格式一:保存成all.txt.parse 然后就可以直接训练了
        # 来源：https://github.com/DRUNK2013/lm-ken
        
        训练过程:
            输入 : self.corpus_path语料文件
            输出 : self.arpa_file语料文件
        
        报错：
        34304 , 需要增加样本量
        
        '''
        #corpus_file,arpa_file,memory = '50%',order = 4,skip_symbols = '"<unk>"'
        executive_code = self.kenlm_model_path + 'lmplz -S {} -o {} --skip_symbols {} < {} > {} '.format(self.memory,self.order,self.skip_symbols,self.corpus_file,self.arpa_file)
        status = os.system(
                    executive_code
                     )
        if status == 0:
            return 'success,code is : %s , \n code is : %s '%(status,executive_code)
        else:
            return 'fail,code is : %s ,\n code is : %s '%(status,executive_code)
    
    def convert_format(self):
        '''
        # 压缩模型
        # 来自苏神：https://spaces.ac.cn/archives/3956#%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83
        
        ```
        ./kenlm/bin/build_binary weixin.arpa weixin.klm
        ```
        
        arpa是通用的语言模型格式，klm是kenlm定义的二进制格式，klm格式占用空间更少。
        
        报错：
        256 ： No such file or directory while opening output/test2.arpa
    
        '''
        #arpa_file,klm_file,memory = '50%'
        executive_code = self.kenlm_model_path + 'build_binary -S {} -s {} {}'.format(self.memory,self.arpa_file,self.klm_file)
        status = os.system(
                    executive_code
                     )
        if status == 0:
            return 'success,code is : %s , \n code is : %s '%(status,executive_code)
        else:
            return 'fail,code is : %s ,\n code is : %s '%(status,executive_code)
    
            
    '''
        分词模块
        主要引用苏神 : 【中文分词系列】 5. 基于语言模型的无监督分词
    '''
    def parse_text(self,text):
        return ' '.join(list(text))
    

    def viterbi(self,nodes):
        '''  # 分词系统
        #这里的转移概率是人工总结的，总的来说，就是要降低长词的可能性。
        #trans = {'bb':1, 'bc':0.15, 'cb':1, 'cd':0.01, 'db':1, 'de':0.01, 'eb':1, 'ee':0.001}
        #trans = {i:log10(j) for i,j in trans.items()}
        
        苏神的kenlm分词:
            b：单字词或者多字词的首字
            c：多字词的第二字
            d：多字词的第三字
            e：多字词的其余部分
        '''
        # py3的写法
        paths = nodes[0]
        for l in range(1, len(nodes)):
            paths_ = paths
            paths = {}
            for i in nodes[l]:
                nows = {}
                for j in paths_:
                    if j[-1]+i in self.trans:
                        nows[j+i]= paths_[j]+nodes[l][i]+self.trans[j[-1]+i]
                #k = nows.values().index(max(nows.values()))
                k = max(nows, key=nows.get)
                #paths[nows.keys()[k]] = nows.values()[k]
                paths[k] = nows[k]
        #return paths.keys()[paths.values().index(max(paths.values()))]
        return max(paths, key=paths.get)
    
    def cp(self,s):
        if self.model == None:
            raise KeyError('please load model(.klm / .arpa).')
        return (self.model.score(' '.join(s), bos=False, eos=False) - self.model.score(' '.join(s[:-1]), bos=False, eos=False)) or -100.0
    
    def mycut(self,s):
        nodes = [{'b':self.cp(s[i]), 'c':self.cp(s[i-1:i+1]), 'd':self.cp(s[i-2:i+1]),\
                  'e':self.cp(s[i-3:i+1])} for i in range(len(s))]
        tags = self.viterbi(nodes)
        words = [s[0]]
        for i in range(1, len(s)):
            if tags[i] == 'b':
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
    
    
    '''
        kenlm n-grams训练模块 + 新词发现
        主要引用苏神的：重新写了之前的新词发现算法：更快更好的新词发现
    '''
    
    def unpack(self,t, s):
        return struct.unpack(t, s)[0]
    
    def read_ngrams(self):
        """读取思路参考https://github.com/kpu/kenlm/issues/201
        """
        # 数据读入
        f = open(self.vocab_file)
        chars = f.read()
        f.close()
        chars = chars.split('\x00')
        chars = [i for i in chars] # .decode('utf-8')
        # 
        ngrams = [Counter({}) for _ in range(self.order)]
        total = 0
        size_per_item = self.order * 4 + 8
        f = open(self.ngram_file, 'rb')
        filedata = f.read()
        filesize = f.tell()
        f.close()
        for i in range(0, filesize, size_per_item):
            s = filedata[i: i+size_per_item]
            n = self.unpack('l', s[-8:])
            if n >= self.min_count:
                total += n
                c = [self.unpack('i', s[j*4: (j+1)*4]) for j in range(self.order)]
                c = ''.join([chars[j] for j in c if j > 2])
                for j in range(self.order):# len(c) -> self.order
                    ngrams[j][c[:j+1]] = ngrams[j].get(c[:j+1], 0) + n
        return ngrams,total
    
    
    def filter_ngrams(self,ngrams, total, min_pmi=1):
        """通过互信息过滤ngrams，只保留“结实”的ngram。
        """
        order = len(ngrams)
        if hasattr(min_pmi, '__iter__'):
            min_pmi = list(min_pmi)
        else:
            min_pmi = [min_pmi] * order
        #output_ngrams = set()
        output_ngrams = Counter()
        total = float(total)
        for i in range(order-1, 0, -1):
            for w, v in ngrams[i].items():
                pmi = min([
                    total * v / (ngrams[j].get(w[:j+1], total) * ngrams[i-j-1].get(w[j+1:], total))
                    for j in range(i)
                ])
                if math.log(pmi) >= min_pmi[i]:
                    #output_ngrams.add(w)
                    output_ngrams[w] = v
        return output_ngrams
    
    '''
        智能纠错模块
            主要与pycorrector互动
    '''
    def is_Chinese(self,word):
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False
    
    def word_match(self,text_a,text_b): 
        '''
        筛选规则:
            # 字符数一致
            # 不为空
            # 拼音首字母一致
            
        输出:
            最佳是否相似,bool
        '''
    
        pinyin_n,match_w = 0,[]
        text_a_pinyin = lazy_pinyin(text_a, style=Style.FIRST_LETTER) 
        text_b_pinyin = lazy_pinyin(text_b, style=Style.FIRST_LETTER) 
        #print(text_a_pinyin,text_b_pinyin)
        if len(text_a) > 0 and (len(text_b)  == len(text_a) ) and self.is_Chinese(text_a) and self.is_Chinese(text_b):
            for n,w1 in enumerate(text_a):
                if text_b[n] == w1:
                    match_w.append(w1)
                if text_a_pinyin[n] == text_b_pinyin[n]:
                    pinyin_n += 1
            return True if len(match_w) > 0 and pinyin_n == len(text_a) else False
        else:
            return False
        
    def compare(self,text_a,text_b): 
        '''
        通过kenlm对比两个文本的优劣:
            text_a - text_b > 0 , text_a 好
        '''
        return self.model.score(' '.join(text_a), bos=False, eos=False) - self.model.score(' '.join(text_b), bos=False, eos=False)
    
    def find_best_word(self,word,ngrams,freqs = 10):
        '''
        通过kenlm找出比word更适合的词
        
        输入:
            word,str
            ngrams,dict,一个{word:freq}的词典
            
        输出:
            最佳替换word
        '''
        candidate = {bg:freq for bg,freq in ngrams.items() if self.word_match(word,bg) &  (freq > freqs) }
        #if len(candidate) == 0:
        #    raise KeyError('zero candidate,large freqs')
        candidate_score = {k:self.compare(k,word) for k,v in candidate.items()}
        if len(candidate_score) > 0:
            return max(candidate_score, key=candidate_score.get)
        else:
            return word

    def word_discovery(self,ngrams_dict,\
                       good_pos = ['n','v','ag','a','zg','d'],\
                       bad_words = ['我','你','他','也','的','是','它','再','了','让'] ):
        '''
        新词筛选
        筛选规则：
            - jieba分不出来
            - 词性也不包括以下几种
    
        jieba词性表：https://blog.csdn.net/orangefly0214/article/details/81391539
    
        坏词性：
            uj,ur,助词
            l,代词
    
        好词性：
            n,v,ag,a,zg,d(副词)
        '''
        new_words_2 = {}
        
        for nw,freq in tqdm(ngrams_dict.items()):
            jieba_nw = list(jieba.cut(nw))
            words = list(pseg.cut(nw))
            pos = [list(wor)[1] for wor in words]
            if( len(jieba_nw) != 1)  and  ( len( [gp for gp in good_pos if gp in ''.join(pos)]  ) > 0  ) and    (  len([bw for bw in bad_words if bw in nw[0]]) == 0   ):
                new_words_2[nw] = freq
                #print(list(words))
        return new_words_2


if __name__ == "__main__":

    '''
        模型训练与生成
        
        这里注意save_path是存放一些生成文件的路径
    '''
    # 模型加载
    km = kenlm_model(save_path = 'output',project = 'test2',\
                     memory = '50%',min_count = 2,order = 4,\
                     skip_symbols = '"<unk>"',kenlm_model_path = './kenlm/build/bin/')
    
    # NLM模型训练文件生成
    text_list = ['Whoo 后 拱辰享水 水妍护肤套装整套质地都比较清爽，滋润，侧重保湿，适合各种肤质，调节肌肤水平衡，它还具有修复功效，提亮肤色我是油性肤质用起来也一点也不觉得油腻，味道淡淡的还很好闻，也很好吸收，质地清爽，上脸没有油腻感，就算是炎炎夏日用起来也相当舒服它的颜值很高的，粉粉的超级少女，满足了老夫的少女心保湿指数⭐️⭐️⭐️⭐️⭐️推荐指数⭐️⭐️⭐️⭐️⭐️如果你是油性肤质，选择水妍系列，再也不会担心每天到了下午顶着一个大油饼子脸了，油性、混合性妹子完全不用担心踩雷性价比超高，这一套满足了日常护肤的基本要求(Whoo洁面膏)',
     '盘点不怕被税的海淘网站❗️海淘向来便宜又保真，比旗舰店、专柜和代购好太多！还能体验海淘乐趣~外网需要双币信用卡，往往需要转运，北上地区容易被税。安利大家几家包税/有税补的海淘商家[赞R]\xa01⃣️beautylish?全中文页面支付宝/微信/银联100%免关税75美元全球包邮相对其他英淘，小众产品比较齐全，RCMA的散粉是无限回购好物，买过他们家福袋感觉蛮超值的。据说去年双十二爆出海关扣了一批货，客服装死。2⃣️Chemistforless?全中文界面支付宝/微信现金税补现金包税不限时免邮活动邮费超重部分按6.5澳/kg，比同类型网站的8澳便宜很多，同产品比较价格相对便宜。购买过三次，最快的一次的当天就出单号了，一周到国内，推荐它家孕妇DHA和蜂蜜面膜，我囤的比较多\xa03⃣️lookfantastic?全中文界面支付宝/微信/银联20英镑税补50英镑包邮网站的每月礼盒是我的最爱，送人必备，便宜又有逼格~只要你看好分单，金额控制在600以下不容易被税，所有网站都要分箱。很重要❗️chemistforless会帮你自动分箱，其他的网站需要自己分单下。\xa04⃣️perfume’s club?全中文界面支付宝/银联/微信20欧税补60欧包邮西班牙美妆网站~从西班牙/香港出货，买的几次都是西班牙直邮~网站品牌齐全，折扣较多，不着急的宝贝还是可以下单的。不过他家被税的几率有点大\xa05⃣️feelunique?全中文界面支付宝/银联/微信20英镑税补60英镑包邮搜索关键词不是很好，要点品牌看，品牌多样，下单时候有小样选择~物流应该走的阳光清关线路，西班牙发出，境外物流查不到，进了国内才能看到物流，FU补货速度一言难尽6⃣️wruru?全中文界面支付宝/微信现金包税1000免邮首家俄罗斯网站，物流走国际申通，比较慢。但俄罗斯的美妆向来很优惠~做活动的时候会送小样，品牌齐全，娇韵诗价格非常nice，暗戳戳觉得这家黑五娇韵诗一定会折上折，可以趁机囤一波双萃~大家口已一起上车～\xa0以上网站提及税补，小宝贝们被税的时候一定要保管好被税凭证，不然税补很困难~被税时不要紧张，金额不是很多老老实实交税吧，包裹退回的运费需自己承担，一来一去等于啥都没买还倒贴运费一般网站的微博和小红书也有活动~想买但没有优惠可以悄咪咪去微博小红书摸一摸优惠券哦[赞R](娇韵诗双萃,娇韵诗面膜)',
     '学生用什么洗面奶好？学生党必备的这六款性价比最高的洗面奶是什么？你用过吗？DreamtimesM2梦幻洁面乳参考价格￥67Dreamtimes是一款专注年轻人护肤的品牌。他们家的东西都适合年轻的肌肤，所以这个品牌也是广大学生党的挚爱。这款洁面乳算是一款爆款产品了，里面含有洁肤因子，有效控油洁肤，是一款复合型洁面乳，对痘痘肌肤和敏感肌都是非常适合的，是一款性价比很高的洁面。妮维雅慕斯洗面奶参考价格￥35妮维雅家的新产品，一款氨基酸慕斯洁面。绵软的泡沫和奶油的质地让让欲罢不能。有效深入肌肤底层清洁肌肤，同时氨基酸配方，低刺激，温和洁面。如果你是敏感和易过敏肌肤，那这款就是你的最佳选择了。悦诗风吟绿茶洗面奶参考价格￥55这款也是很适合学生党的一款平价大碗洗面奶，里面蕴含绿茶精华，有效舒缓痘痘肌肤和改善油性肌肤等问题，绿茶精华可以舒缓痘痘肌肤，缓解红肿、过敏等症状。泡沫丰富细腻，改善出油状况的同时可以缓解痘痘肌肤。(妮维雅洗面奶,悦诗风吟洗面奶氨基酸)',
     '国货大宝。启初。……使用分享修复：玉泽或至本（第三代玉泽身体乳没有了麦冬根和神经酰胺）。芦荟胶（含酒精，不锁水，偶尔敷一下，皮肤会越用越干）。Swisse蜂蜜面膜（清洁鼻子，效果肉眼可见，不能常用）。大宝sod别用在脸上，会变黑，脸疼，当护手霜，以后不买了。迷奇套装（含很多香精，可能只适合油性皮肤），香精可能会导致敏感。启初面霜（平价珂润，但珂润有神经酰胺），但好像没什么效果，油痘皮最好不要用，以后不要买了。(玉泽身体乳,珂润身体乳,珂润护手霜,玉泽面霜)',
     '资生堂悦薇乳液，会回购。夏天用略油腻，冬天用刚好。真的有紧致感，28岁，眼部有笑纹，其他地方还可以。这是第二个空瓶。冬天会回购。没有美白效果。(资生堂悦薇)']
    
    km.write_corpus(km.text_generator(text_list,jieba_cut = False), km.corpus_file) # 将语料转存为文本
    
    # NLM模型训练
    status = km.lm_train()
    print(status)
    
    # NLM模型arpa文件转化
    km.convert_format()
    
    
    '''
        新词发现
    '''
    # 模型n-grams生成
    km.count_ngrams()
    
    # 模型读入与过滤
    ngrams,total = km.read_ngrams()
    ngrams_2 = km.filter_ngrams(ngrams, total, min_pmi=[0, 1, 3, 5])
    
    # 新词发现
    km.word_discovery(ngrams_2)
    
    
    '''
        智能纠错
    '''
    # 加载模型
    km.model = km.load_model(km.klm_file)
    
    # n-grams读入
    ngrams,total = km.read_ngrams()
    ngrams_2 = km.filter_ngrams(ngrams, total, min_pmi=[0, 1, 3, 5])
    
    # 纠错
    import pycorrector
    from pypinyin import lazy_pinyin, Style
    sentence = '这瓶洗棉奶用着狠不错'
    idx_errors = pycorrector.detect(sentence)
    
    correct = []
    for ide in idx_errors:
        right_word = km.find_best_word(ide[0],ngrams_2,freqs = 0)
        if right_word != ide[0]:
            correct.append([right_word] + ide)
    
    print('错误：',idx_errors)
    print('pycorrector的结果：',pycorrector.correct(sentence))
    print('kenlm的结果：',correct)
