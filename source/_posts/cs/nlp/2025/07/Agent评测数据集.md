---
title: Agent评测数据集
tags:
  - NLP
  - LLM
  - Agent
categories:
  - CS
  - NLP
  - Agent
abbrlink: 72150a83
date: 2025-07-06 22:03:18
---

整合一下agent常用的评测数据集。（虽然评测数据集很重要，但是谨记Goodhart's law，迷信测试指标也不可取。）  

# overview  

| 数据集 | 语言 | 难度 | 数量 | 模态 | 领域 | 评测方式 |
| --- | --- | --- | --- | --- | --- | --- |
|GAIA|英文|高|166dev+300test|多模态|涵盖个人日常任务，科学问题，以及通用信息查询|可自动化|
|BrowseComp|英文|高|1266|文本|多领域|可自动化
|BrowseComp-ZH|中文|高|289|文本|11个领域|可自动化
|HLE|英文|高|2700+|13%多模态问题|数学、人文科学、自然科学等数十个不同的学科|可自动化|
|GPQA|英语|中|448道多选题|文本|生物学、物理学和化学|可自动化|
|ScholarSearch|英文|中|223|文本|覆盖Science & Engineering和Social Sciences & Humanities两大门类，共15个细分学科|可自动化|

# GAIA  

数据集：GAIA = General AI Agent Assessment，由 Meta、HuggingFace 等团队联合提出。  

数据内容：GAIA所有问题都是研究团队设计的，问题语言都是英文，参考答案也是英文（因此也很依赖英文的搜索工具，所以在中文场景可能略有些局限性）。数据中，大部分是文本（约70%），小部分涉及视频、音频、图像和excel、pdf、ppt、png、jpg、csv、pdb、mp3等多种模态和格式。共有466条数据，其中166条包含唯一答案的问题（val set），因此很适合自动化评测（比如LLM-as-judge）；另外还有300条不包含答案的数据（test set，用于leaderboard排名）。  

GAIA的数据考察Agent多个维度的能力，包括：推理能力、多模态能力、网页浏览&信息检索、工具使用能力、世界知识等。  

GAIA把问题根据难度分成3个等级：level 1，level 2和level 3。level 1最简单，level 3最难，难度越高，模态也越丰富。通常来说难度越高，需要的工具和步骤越多（但也不是严格绝对的）。  

放几条数据样例（val set）的感受一下，数据里除了问题和答案，还有中间step：  

```python
{"task_id": "c61d22de-5f6c-4958-a7f6-5e9707bd3466", "Question": "A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?", "Level": 2, "Final answer": "egalitarian", "file_name": "", "Annotator Metadata": {"Steps": "1. Go to arxiv.org and navigate to the Advanced Search page.\n2. Enter \"AI regulation\" in the search box and select \"All fields\" from the dropdown.\n3. Enter 2022-06-01 and 2022-07-01 into the date inputs, select \"Submission date (original)\", and submit the search.\n4. Go through the search results to find the article that has a figure with three axes and labels on each end of the axes, titled \"Fairness in Agreement With European Values: An Interdisciplinary Perspective on AI Regulation\".\n5. Note the six words used as labels: deontological, egalitarian, localized, standardized, utilitarian, and consequential.\n6. Go back to arxiv.org\n7. Find \"Physics and Society\" and go to the page for the \"Physics and Society\" category.\n8. Note that the tag for this category is \"physics.soc-ph\".\n9. Go to the Advanced Search page.\n10. Enter \"physics.soc-ph\" in the search box and select \"All fields\" from the dropdown.\n11. Enter 2016-08-11 and 2016-08-12 into the date inputs, select \"Submission date (original)\", and submit the search.\n12. Search for instances of the six words in the results to find the paper titled \"Phase transition from egalitarian to hierarchical societies driven by competition between cognitive and social constraints\", indicating that \"egalitarian\" is the correct answer.", "Number of steps": "12", "How long did this take?": "8 minutes", "Tools": "1. Web browser\n2. Image recognition tools (to identify and parse a figure with three axes)", "Number of tools": "2"}}
{"task_id": "17b5a6a3-bc87-42e8-b0fb-6ab0781ef2cc", "Question": "I\u2019m researching species that became invasive after people who kept them as pets released them. There\u2019s a certain species of fish that was popularized as a pet by being the main character of the movie Finding Nemo. According to the USGS, where was this fish found as a nonnative species, before the year 2020? I need the answer formatted as the five-digit zip codes of the places the species was found, separated by commas if there is more than one place.", "Level": 2, "Final answer": "34689", "file_name": "", "Annotator Metadata": {"Steps": "1. Search the web for \u201cfinding nemo main character\u201d.\n2. Note the results, which state that the main character is a clownfish.\n3. Search the web for \u201cusgs nonnative species database\u201d.\n4. Click result for the Nonindigenous Aquatic Species site.\n5. Click \u201cMarine Fishes\u201d.\n6. Click \u201cSpecies List of Nonindigenous Marine Fish\u201d.\n7. Scroll through the list until I find the clown anenomefish, and click \u201cCollection info\u201d.\n8. Note the place that a clown anenomefish was found, in Fred Howard Park at the Gulf of Mexico.\n9. Search the web for \u201cfred howard park florida zip code\u201d.\n10. Note the zip code, 34689. Since only one clownfish was found before the year 2020, this is the answer.", "Number of steps": "10", "How long did this take?": "5 minutes", "Tools": "1. Search engine\n2. Web browser", "Number of tools": "2"}}
{"task_id": "04a04a9b-226c-43fd-b319-d5e89743676f", "Question": "If we assume all articles published by Nature in 2020 (articles, only, not book reviews/columns, etc) relied on statistical significance to justify their findings and they on average came to a p-value of 0.04, how many papers would be incorrect as to their claims of statistical significance? Round the value up to the next integer.", "Level": 2, "Final answer": "41", "file_name": "", "Annotator Metadata": {"Steps": "1. Find how many articles were published in Nature in 2020 by Googling \"articles submitted to nature 2020\"\n2. Click through to Nature's archive for 2020 and filter the results to only provide articles, not other types of publications: 1002\n3. Find 4% of 1002 and round up: 40.08 > 41", "Number of steps": "3", "How long did this take?": "5 minutes", "Tools": "1. search engine\n2. calculator", "Number of tools": "2"}}
{"task_id": "14569e28-c88c-43e4-8c32-097d35b9a67d", "Question": "In Unlambda, what exact charcter or text needs to be added to correct the following code to output \"For penguins\"? If what is needed is a character, answer with the name of the character. If there are different names for the character, use the shortest. The text location is not needed. Code:\n\n`r```````````.F.o.r. .p.e.n.g.u.i.n.si", "Level": 2, "Final answer": "backtick", "file_name": "", "Annotator Metadata": {"Steps": "1. Searched \"Unlambda syntax\" online (optional).\n2. Opened https://en.wikipedia.org/wiki/Unlambda.\n3. Note that the hello world program is very similar in syntax to the code in this question.\n4. Go to the source referenced by the hello world program.\n5. From the referenced source, read what the components of the program do to understand that each period needs a backtick after the initial `r.\n6. Observe that in the given code, there are 12 periods but only 11 backticks after the initial `r, so the missing character is a backtick.", "Number of steps": "6", "How long did this take?": "15 minutes", "Tools": "1. Web browser\n2. Search engine\n3. Unlambda compiler (optional)", "Number of tools": "3"}}
{"task_id": "e1fc63a2-da7a-432f-be78-7c4a95598703", "Question": "If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary.", "Level": 1, "Final answer": "17", "file_name": "", "Annotator Metadata": {"Steps": "1. Googled Eliud Kipchoge marathon pace to find 4min 37sec/mile\n2. Converted into fractions of hours.\n3. Found moon periapsis in miles (225,623 miles).\n4. Multiplied the two to find the number of hours and rounded to the nearest 100 hours.", "Number of steps": "4", "How long did this take?": "20 Minutes", "Tools": "1. A web browser.\n2. A search engine.\n3. A calculator.", "Number of tools": "3"}}
{"task_id": "32102e3e-d12a-4209-9163-7b3a104efe5d", "Question": "The attached spreadsheet shows the inventory for a movie and video game rental store in Seattle, Washington. What is the title of the oldest Blu-Ray recorded in this spreadsheet? Return it as appearing in the spreadsheet.", "Level": 2, "Final answer": "Time-Parking 2: Parallel Universe", "file_name": "32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx", "Annotator Metadata": {"Steps": "1. Open the attached file.\n2. Compare the years given in the Blu-Ray section to find the oldest year, 2009.\n3. Find the title of the Blu-Ray disc that corresponds to the year 2009: Time-Parking 2: Parallel Universe.", "Number of steps": "3", "How long did this take?": "1 minute", "Tools": "1. Microsoft Excel", "Number of tools": "1"}}
{"task_id": "8e867cd7-cff9-4e6c-867a-ff5ddc2550be", "Question": "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.", "Level": 1, "Final answer": "3", "file_name": "", "Annotator Metadata": {"Steps": "1. I did a search for Mercedes Sosa\n2. I went to the Wikipedia page for her\n3. I scrolled down to \"Studio albums\"\n4. I counted the ones between 2000 and 2009", "Number of steps": "4", "How long did this take?": "5 minutes", "Tools": "1. web browser\n2. google search", "Number of tools": "2"}}
{"task_id": "3627a8be-a77f-41bb-b807-7e1bd4c0ebdf", "Question": "The object in the British Museum's collection with a museum number of 2012,5015.17 is the shell of a particular mollusk species. According to the abstract of a research article published in Science Advances in 2021, beads made from the shells of this species were found that are at least how many thousands of years old?", "Level": 2, "Final answer": "142", "file_name": "", "Annotator Metadata": {"Steps": "1. Use search engine to search for \"British Museum search collection\" and navigate to the British Museum's collection search webpage.\n2. Select \"Museum number\" as search field and \"2012,5015.17\" in text box, then run search.\n3. Open the page for the single result and note that the description says that this is the shell of an individual of the Nassa gibbosula species.\n4. Use search engine to search for \"Nassa gibbosula\".\n5. Note that according to the search result from the World Register of Marine Species website, Nassa gibbosula is not an accepted species name.\n6. Open the page for Nassa gibbosula on the World Register of Marine Species website.\n7. Scan the page and note that the accepted species name is Tritia gibbosula.\n8. Use search engine to search for \"Science Advances 2021 Tritia gibbosula\".\n9. Find that the top result is an article from 2021 in Science Advances titled \"Early Middle Stone Age personal ornaments from Bizmoune Cave, Essaouira, Morocco\".\n10. Scan abstract and note that the article discusses beads made from Tritia gibbosula shells that date to at least 142 thousand years ago, giving a final answer of 142.", "Number of steps": "10", "How long did this take?": "12 minutes", "Tools": "1. Web browser\n2. Search engine", "Number of tools": "2"}}
```

可以看到题目的难度是很大的，人来操作也要费一些功夫，特别是level 2和level 3的题目。  

比如第一题，level 2的：“一篇关于人工智能监管的论文最初于2022年6月提交到arXiv.org，其中展示了一个包含三个坐标轴的图表，每个坐标轴的两端都有标签词。在2016年8月11日提交到arXiv.org的《物理与社会》文章中，这些词中哪一个被用来描述一种社会类型？”  

再看一个，也是level 2的：“我正在研究那些被人们当作宠物饲养后放生并成为入侵物种的生物。有一种鱼类因作为电影《海底总动员》的主角而成为热门宠物。根据美国地质调查局（USGS）的数据，在2020年之前，这种鱼在哪些地方作为非本地物种被发现？答案需要以发现地的五位邮政编码格式呈现，若有多处发现地则用逗号分隔。”  

看看level 1的：“如果埃鲁德·基普乔格能够无限保持他创造纪录的马拉松配速，那么他以最近距离从地球跑到月球需要多少千小时？计算时请使用维基百科上月球的最小近地点数值，并将结果四舍五入到最接近的1000小时，且不要使用任何逗号分隔符。”  

最后看个level 3的感受一下：“梅赛德斯·索萨在2000年至2009年（含）期间发行了多少张录音室专辑？你可以使用2022年最新版本的英文维基百科。”  

# BrowseComp  

BrowseComp（Browsing Competition），由 OpenAI 开源，用于评估agent网络浏览能力的基准测试数据集。  

共包含1266个问题，涵盖多个领域。BrowseComp都是文本，没有多模态数据。而且答案都设计为简短、明确的答案，因此可以做自动化验证。  

样例：  

1、足球比赛查询  

问题："Identify the two football teams that played a match officiated by a Brazilian referee between 1990 and 1994 where there were four yellow cards (two for each team), three of which were not issued in the first half, and there were four substitutions, one of which was an injury substitution before the 25th minute of the game."  

答案：爱尔兰对罗马尼亚 (Ireland vs. Romania)  

2、学术论文查找  

问题："Find a research paper published before June 2023 that mentions cultural traditions, scientific processes, and culinary innovation, and is co-authored by three individuals, one of whom is an Assistant Professor from West Bengal and another holds a PhD."  

答案：《面包制作的基础：面包的科学》(The Fundamentals of Bread Making: The Science of Bread)  

# BrowseComp-ZH  

BrowseComp的中文版，由港科大（广州）、北京大学、浙江大学、阿里巴巴、字节跳动、NIO 等联合发布。大致上是参考BrowseComp搞的。  

共有289条数据，覆盖11个领域。  

样例：  

example 1：  

```
话题：艺术
问题：在中国传统艺术中，有一种特殊的绘画形式，起源于元代，盛行于清末，传说是由一位古代知名画家在酒后兴起所创。这种艺术形式在2010~2015年之间被列入某省级非物质文化遗产名录。绘制这种艺术形式需要画家精通各种画法，并且要善书各种字体。请问这种艺术形式是什么？
答案：锦灰堆
```

example 2：  

```
话题：影视
问题：某知名电视剧，女二号（演员）在1993年进入演艺圈。女一号（演员）的现任丈夫是浙江湖州人。男一号（演员）6年后登上了春晚舞台。问该电视剧是什么？
答案：父母爱情
```

# HLE  

Humanity's Last Exam，人类的最后考试，覆盖了数学、人文科学、自然科学等数十个不同的学科领域。  

基本信息：  

- 题目数量：2700+  
- 多模态数据：占比13%  
- 多选题占比：24%  
- 精确匹配题占比：76%  

领域分布：  

{% asset_img hle_domain.png agent评测 %}  

模型表现对比：  

- OpenAI Deep Research：26.6% 准确率  
- Kimi-Researcher：26.9% 准确率  
- DeepSeek-R1：9.4% 准确率  
- Gemini 2.5 Pro：21.6% 准确率  

样例：  

{% asset_img hle_1.png agent评测 %}  

{% asset_img hle_2.png agent评测 %}  

{% asset_img hle_3.png agent评测 %}  

# GPQA  

Graduate-Level Google-Proof Q&A。  

主要包括大学研究生级别的生物学、物理学和化学题目，共有448道多选题：  

- 生物学：33%，约148题  
- 物理学：35%，约157题  
- 化学：32%，约143题  

测评结果：  

- 领域博士专家：65% 正确率  
- 高技能非专家：34% 正确率  
- GPT-4 基线模型：39% 正确率  

样例：  

{% asset_img gpqa_1.png agent评测 %}  

{% asset_img gpqa_2.png agent评测 %}  

# ScholarSearch  

ScholarSearch 是由北京大学 DS-Lab 团队发布的首个专门用于评估大语言模型在学术研究场景下复杂信息检索能力的数据集，包含223道高难度的学术检索题目及其对应的标准答案。  

ScholarSearch是纯文本数据集，覆盖Science & Engineering和Social Sciences & Humanities两大门类，共15个细分学科。数据集主要英文为主。  

GPT-4o-search-preview的得分为18.83%。  

样例：  

```
Domain: Physics

Question: The State Key Laboratory of Magnetism at the Institute of Physics, Chinese Academy of Sciences/Beijing National Research Center for Condensed Matter Physics recently published groundbreaking research. They used the antiferromagnetic insulator LaFeO3 (LFO) as an electron donor to construct LNOn/LFO1 (n=1-5) superlattices. By modulating the charge transfer at the interface (Fe³⁺→Ni³⁺), they controlled the orbital hybridization degree of the Fe-ONi band, inducing robust ferromagnetism well above room temperature in the 1:1 superlattice, i.e., the layered double perovskite La2NiFeO6 thin film. In the experimental results of this paper, what is the Curie temperature (in Kelvin) of the 3:1 superlattice LaNiO3/LaFeO3?

Answer: 589K
Explanation: Search for "Chinese Academy of Sciences publishes research on inducing high-temperature ferromagnetism in La2NiFeO6 double perovskite thin films" to identify the paper title as Ferromagnetism in LaFeO3/LaNiO3 Superlattices with High Curie Temperature. Locate the paper via Google Scholar and refer to the experimental results section to confirm the answer as 589K.
```

```
Domain: Economics

Question: China's accounting standards are revised annually. Compared to the 2023 accounting standards, what direct change was made in the 2024 accounting standards under CAS 22 – Recognition and Measurement of Financial Instruments regarding the subsequent measurement of financial assets at amortized cost, specifically in the accounting treatment of accrued interest?

Answer: The "Interest Receivable" and "Interest Payable" accounts were changed to "Debt Investments – Accrued Interest."

Explanation: After searching multiple web pages and referring to https://finance.sina.cn/2024-06-11/detail-inaykqrw3776358.d.html, Table 10 (Section 2), the 2024 revision of CAS 22 – Recognition and Measurement of Financial Instruments adjusted the accounting treatment as follows: For debt investments, other debt investments, trading financial assets (bonds), bonds payable, and trading financial liabilities (bonds) with periodic interest payments, the year-end accrual of coupon interest was changed from the "Interest Receivable/Payable" accounts to the "Debt Investments – Accrued Interest" sub-account.
```

***  

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)
博主微信号(添加请注明来意)：  
![](/images/wechat.png)  

***  

【推荐文章】  
- Agent：  
[Agent完全手册(零)：三大模块，三个理念](https://www.linsight.cn/b242bfb3.html)  
[DeepResearch的报告生成方法](https://www.linsight.cn/44c62dc5.html)  
[从RAG到DeepSearch](https://www.linsight.cn/7c2f9dcb.html)  
[agent调研(1)--MetaGPT,OpenManus和OWL](https://www.linsight.cn/226b059f.html)  
[Devin和Anthropic的Agent开发经验](https://www.linsight.cn/f93b3aaf.html)  
- MoE：  
[DeepSeek-V3细节探索](https://www.linsight.cn/a9c496e3.html)  
[MoE模型的前世今生](http://www.linsight.cn/44e38c1b.html)  
[DeepSeek-V2和MLA](https://www.linsight.cn/83c49df0.html)  
[昆仑万维-SkyworkMoE](https://www.linsight.cn/1d5bcd45.html)  
[成本10w刀的JetMoE](https://www.linsight.cn/f3acf042.html)  
[MoE的top-p routing](https://www.linsight.cn/224c42da.html)  
[对MoE模型的一些观察](https://www.linsight.cn/5e1d14b3.html)  
[从dense到MoE -- sparse upcycling](https://www.linsight.cn/a0824e29.html)  
[MoE路由--expert choice routing](https://www.linsight.cn/2c8bbc7.html)  
- 端侧模型：  
[苹果智能系统模型--AFM](https://www.linsight.cn/1e34e252.html)  
[MiniCPM](https://www.linsight.cn/376db710.html)  
[适合移动设备的语言模型--MobileLLM](https://www.linsight.cn/5ac36d34.html)  
[phi系列模型](https://www.linsight.cn/fe13b56f.html)  
[Gemma2](https://www.linsight.cn/cf3f1f81.html)  
[苹果的OpenELM](https://www.linsight.cn/f845f3e4.html)  
[bilibili的index-1.9B](https://www.linsight.cn/770b63e1.html)  
- 预训练：  
[Qwen3实测&技术报告](https://www.linsight.cn/37ee84bb.html)  
[代码大模型(一)--业界现状](https://www.linsight.cn/a0b50049.html)  
[代码大模型(二)--OpenCoder](https://www.linsight.cn/7856bcc1.html)  
[LLM高效预训练(一)](https://www.linsight.cn/dcb57672.html)  
[LLM高效预训练(二)](https://www.linsight.cn/1e2e35a7.html)  
[Llama3.1--预训练要点一览](https://www.linsight.cn/7d7294cb.html)  
[Qwen2技术报告](https://www.linsight.cn/a8f8b641.html)  
[Yi技术报告-划重点看细节](http://www.linsight.cn/41b6a819.html)  
[InternLM系列模型](https://www.linsight.cn/7f3d361.html)  
[GLM4报告的一些技术点](https://www.linsight.cn/a5206abd.html)  
[从Yuan2.0到Yuan2.0-M32](https://www.linsight.cn/3df0cd42.html)  
[从loss视角理解大模型涌现能力](https://www.linsight.cn/f5fb75e4.html)  
- 数据：  
[训练数据合成(一)](https://www.linsight.cn/85132189.html)  
[训练数据合成(二)](https://www.linsight.cn/2a22baeb.html)  
[训练数据合成(三)](https://www.linsight.cn/e259c7b2.html)  
[LLM预训练数据策略(一)](https://www.linsight.cn/2c2cdc34.html)  
[预训练数据处理--长度分解](https://www.linsight.cn/210dbccd.html)  
- 长上下文：  
[Qwen2.5-1M技术解析](https://www.linsight.cn/6c0f6207.html)  
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
[prompt压缩(一)](https://www.linsight.cn/4519eadd.html)  
[prompt压缩(二)](https://www.linsight.cn/ea2871bf.html)  
[reasoning压缩(一)](https://www.linsight.cn/bfa4f144.html)  
- 推理加速：  
[大模型推理加速-投机解码](http://www.linsight.cn/f5c015c.html)  
[大模型推理加速-MEDUSA](https://www.linsight.cn/7bbe2df6.html)  
- 对齐：  
[深度求索DeepSeek-R1详解](https://www.linsight.cn/9e4b4e6d.html)  
[基模型Cognitive Behaviors对RL的影响](https://www.linsight.cn/657a6d17.html)  
[Llama3.1--post-training要点一览](https://www.linsight.cn/93328a2a.html)  
[模型平均 -- model soup](https://www.linsight.cn/bb8fcf21.html)  
[大模型偏好对齐-DPO](http://www.linsight.cn/473f2b43.html)  
[大模型偏好对齐-ODPO](http://www.linsight.cn/da871ebe.html)  
[大模型偏好对齐-simPO](http://www.linsight.cn/280fa97a.html)  
[大模型偏好对齐-IPO](http://www.linsight.cn/4fe7b810.html)  
- Transformer：  
[理解Attention:从起源到MHA,MQA和GQA](http://www.linsight.cn/3dc22f96.html)  
[LLM的重复生成和ICL](https://www.linsight.cn/7381cae3.html)  
[transformer中normalization的二三事](http://www.linsight.cn/6a40bfa5.html)  
[从代码实现看normalization-到底做了什么](http://www.linsight.cn/b70b4a2d.html)  
[稀疏注意力计算:sliding window attention](http://www.linsight.cn/c61d17e3.html)  
[理解LLM位置编码:RoPE](http://www.linsight.cn/a051710f.html)  
[RoPE的远距离衰减](https://www.linsight.cn/f0902f1a.html)  
[LLM水印](https://www.linsight.cn/2dee4921.html)  
- 训练框架  
[LLM训练框架：从优化器和精度讲到ZeRO](https://www.linsight.cn/fe0adaa5.html)  
[LLM训练各种并行策略](https://www.linsight.cn/4cd8532f.html)  
- 项目应用：  
[一个模型支持智能助手系统](https://www.linsight.cn/9c593ccd.html)  
[关于The Bitter Lesson](https://www.linsight.cn/d253d7b3.html)  
- CV：  
[CV入门--关于Vision Transformer](https://www.linsight.cn/a11e2633.html)  
[CV入门--无监督学习](https://www.linsight.cn/ae81a87b.html)  
- 多模态：  
[多模态入门(一)--CLIP](https://www.linsight.cn/3069051d.html)  
[多模态入门(二)--Flamingo,LLaVA系列和BLIP系列](https://www.linsight.cn/569d722c.html)  
[多模态入门(三)--MiniGPT4,DeepSeekVL,InternVL系列和QwenVL系列](https://www.linsight.cn/f16505b3.html)  
[多模态入门(四)--CogVLM,VILA,MM1,MM1.5和Pixtral-12B](https://www.linsight.cn/e00debee.html)  
[多模态入门(五)--InternVL系列](https://www.linsight.cn/52c8a4f9.html)  
[小米的移动UI多模态模型--MobileVLM](https://www.linsight.cn/96393d3b.html)  
[DeepSeek-VL2的细节](https://www.linsight.cn/b4d047c1.html)  
- 论文阅读：  
[最近阅读--关于数据合成、agent、reasoning和多任务](https://www.linsight.cn/e96c7aac.html)  
[最近阅读2-关于自适应深度思考、context engineering和模型训练](https://www.linsight.cn/af7f9363.html)  
- 大模型算法题：  
[(1)](http://www.linsight.cn/3345028a.html)、
[(2)](http://www.linsight.cn/ad0bba9d.html)、
[(3)](http://www.linsight.cn/1736008.html)、
[(4)](http://www.linsight.cn/1736008.html)、
[(5)](http://www.linsight.cn/336f2f3e.html)、
[(6)](http://www.linsight.cn/7c04944d.html)、
[(7)](https://www.linsight.cn/dd614e12.html)、
[(8)](https://www.linsight.cn/e287b9c3.html)、
[(9)](https://www.linsight.cn/fb9c8882.html)  
