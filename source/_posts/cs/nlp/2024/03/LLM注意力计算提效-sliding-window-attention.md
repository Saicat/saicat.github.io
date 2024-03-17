---
title: '稀疏注意力计算:sliding window attention'
tags:
  - NLP
  - LLM
  - transformer
  - attention
  - sliding window attention
  - sparse attention
categories:
  - CS
  - NLP
  - LLM
abbrlink: c61d17e3
date: 2024-03-12 17:26:00
---

【本文已在同名微信公众号/知乎/个人博客同步上线】  

LLM的长文本能力现在已经是各个大模型巨头的必争之地。  

我们之前在[《LLM长上下文的问题》](http://www.linsight.cn/c4da56c0.html)简单介绍了目前把大模型理解和生成能力推广到32k+/128k+的主流方法，在[《理解Attention:从起源到MHA,MQA和GQA》](http://www.linsight.cn/3dc22f96.html)一文中也解析了MQA和GQA通过节省KV缓存的方式，支持模型在长上下文情况下推理加速的方案。  

在这讲一下另一种（理论有损）提升注意力计算效率的方法：SWA（sliding window attention）。  

一些效果受到广泛关注的模型，如Qwen系列和Mistral就使用了SWA。  

关于Mistral：  

Mistral AI是法国一家AI独角兽公司，2023年5月才成立，但是在2023年9月和12月就分别推出了Mistral 7B和MoE模型Mistral 8x7B并开源。  

2024年2月，微软也投资了它。  

{% asset_img ms_invest_mistral.png MS %}  

它在2024年2月发布的Mistral Large，支持多语言 & 32k的上下文长度，在MMLU上也是获得了直逼GPT4的效果  

{% asset_img mistral_large_performance.jpeg Mistral Large MMLU Performance %}  

（大家也因此对Mistral寄予了厚望，希望它能成为大模型行业的鲶鱼，激活一下OPENAI和META加速一下开源。）  

# SWA

虽然SWA的思路最早不是Mistral提出的，我们还是先以Mistral 7B为例来看下SWA的具体做法。  

## Mistral 7B

2023年10月，Mistral发布了Mistral 7B的[技术报告](https://arxiv.org/pdf/2310.06825.pdf)。其中开篇就说到，相比Llama，Mistral在结构上做了一些改动，除了GQA，另一个用于支持长文本下高效推理的改动就是SWA。  

来看下Mistral 7B的模型结构参数  

{% asset_img mistral_architechture.png Mistral Architechture %}  

Mistral使用了kv组数=8的GQA，intermediate size相比Llama2（11008）大一些，其他基本没有太大变化。  

## 计算量和缓存

对于原始的causal attention，其注意力矩阵是一个下三角矩阵，这样每个token都能看到自己和在自己前面的token。  

这样随着输入长度 $s$ 增大，这个下三角矩阵中1的元素数量以 $s^2$ 的速度增长，带来的是计算量和所需的KV Cache以平方的速度增长。  

（我们知道计算量/缓存和长度 $s$ 成平方关系，这里放一些更具体的推算细节，已经熟悉的朋友可以跳过）

（1）计算量

对于两个这样大小的矩阵相乘： $[m,n]\times[n,p]$ ，输出矩阵大小为 $[m,p]$，共有 $m\times p$ 个元素，每个元素需要 $n$ 次乘法和 $n$ 次加法，因此一次矩阵乘法有 $2mpn$ 个floating point operations（FLOPs）。  

计算量上，按[《Training Compute-Optimal Large Language Models》](https://arxiv.org/pdf/2203.15556.pdf)的算法来。  

对于一般MHA，输入长度为 $s$ ，层数为 $L$ ，模型hidden size为 $d_{model}$ ，每个头的维度为 $d_{q}$ ， 头的数量为 $n_{q}$（这里假设有 $d_{model} = n_{q}\times d_{q}$ ），各个operation的FLOPs如下  

<center>

| Operation | FLOPs（MHA） |
| :---- | :----: |
| Attention: QKV | $6\times s\times h_{model}^{2}$  |
| Attention: QK logits ( $QK^T$ ) | $2\times s^2\times h_{model}$ |
| Attention: Softmax | $3\times n_{q}\times s^2$ |
| Attention: Reduction (apply to $V$) | $2\times s^2\times h_{model}$ |
| Attention: Outupt Linear Project | $2\times s\times h_{model}^{2}$ |

</center>

Softmax项中，对一个 $[1,s]$ 的向量做softmax，计算量为 $3s$ （一个 $s$ 是算每个元素的exp，一个 $s$ 是求和算分母，一个 $s$ 是算除法），而对 $[s,s]$ 的矩阵做softmax，则计算量为  $3s^2$ ，每个头都要计算一遍，因此再乘以 $n_{q}$ 。

（这里忽略了其他一些operation，比如scaling，dropout等，有兴趣的朋友可以自己推算一下）

顺便算下对于Mistral 7B这样使用了GQA的情况。  

其实只有第一项的KV有变化，其他都没变。假设kv头的数量为 $n_{kv}$，则有

<center>

| Operation | FLOPs（GQA） |
| :---- | :----: |
| Attention: QKV | $2\times s\times h_{model}^{2}\\+4\times s\times h_{model}\times (h_{q}\times n_{kv})$  |

</center>

从上面的推算可以看到QK logits、Softmax和Reduction三项是和长度 $s$ 成平方关系的，其他则是线性关系。  

（2）缓存

KV Cache需要缓存的参数量为  

$$
2\times L\times s\times d_{q}\times n_{kv}
$$  

如果使用的是半精度浮点数，那么总共所需的空间就是

$$
2\times 2\times L\times s\times d_{q}\times n_{kv}
$$  

对于Mistral 7B，在输入长度为16k的情况下，所需的KV_Cache约为2G。  

看来虽然用了GQA，但是在长文本（16k+）的情况下计算量和显存还是颇有压力。

## SWA思路

看来要提升attention计算效率，需要想办法减小上面推算中的 $s$ ，但是怎么在减小 $s$ 的同时，还能保持模型长上下文的理解和生成能力呢？

来看一下，CNN中的感受野  

{% asset_img receptive_field_cnn.png CNN Receptive Field %}  

如上图，假设模型有3层，每层卷积核大小为 $3\times 3$ （实际上CNN里卷积操作就是一个sliding window）。  

那对于layer 3，每一个像素能看到layer 2中的一个 $3\times 3$ 的区域，layer 2中其他较远的像素就看到不了。  

但我们再往前推，layer 2里的每个像素也可以看到layer 1中的一个 $3\times 3$ 区域，那么layer 2中的 $3\times 3$ 区域就可以看到layer 1中一个 $5\times 5$ 的区域，相当于layer 3中一个像素可以<u>**间接**</u>看到一个 $5\times 5$ 的输入。  

以此类推，如果我们再增加一层layer 4，那么layer 4中一个像素就能获取输入层（layer 1） 一个 $7\times 7$ 区域的信息。  

虽然每层只能多看周围一格的信息，但是只要我们层数够多，理论上靠近输出端的层想看多远就能看多远。  

值得注意的一点是，我们一般认为模型低层部分提取比较基础的特征，而高层会提取高级的语义特征。  

在CNN里，前几层提取的可能更多是关于简单的边界、颜色、形状等基础特征，而后面的层则提取较复杂的语义特征，比如在分类任务中会是和分类类别相关的花纹、物体大小、风格等特征。  

如果我们把模型设计成，最后一层的一个像素刚好要到第一层才能接收到全局信息（在其它层都只能看到局部），那对于图像边缘的语义特征识别能力可能会受到一些限制。  

具体来说，假设我们做猫和狗的图像分类任务，如果这个时候决定性的特征出现在图像最边缘几个像素里，那这种情况下的错误率会比特征出现在图像中间时要高。  

而对于语言模型，一般情况下，越远距离的信息，对当前位置的重要性越低，因此只要我们的窗口大小不要太过极限小，问题应该都还不大。

看下Mistral的SWA具体是怎么做的  

{% asset_img mistral_swa.png Mistral SWA %}  

左边是正常的causal attention，每个位置能看到自己和前面的位置，attention mask是个下三角矩阵。  

中间则是SWA的attention mask，这里的窗口大小为3。包括自己在内，每个位置只能往前看3个输入。  

同CNN的感受野一样，随着层数的堆叠，模型理论上能处理的最远距离也逐层线性递增。只是LLM里递增的方向是单向的，只能往前。  

Mistral 7B使用了4096的窗口大小，模型层数为32，则最终输出的”感受野“为 $4096\times 32=131,072$ 达到131k的长度。  

前面我们推算了attention的计算量，其中QK logits、Softmax和Reduction三项是和长度 $s$ 成平方关系。在使用了SWA之后，理论上，这几个operation仅使用4k的计算量，就能获得131k的上下文效果。当输入长度为131k时，除去已经缓存部分的数值，新的输入计算量相差 $32\times 32=1024$ 倍。  

而缓存和上下文长度 $s$ 成线性关系，当上下文长度为131k时，最大也能节省 $31/32$ 的显存。  

即SWA在上下文长度在4k以下时，和普通causal attention一样；当上下文长度超过4k时，则相对节省资源，长度越大，节省的比例越高。

>In practice, for a sequence length of 16K and W = 4096, changes made to FlashAttention [11] and xFormers [18] yield a 2x speed improvement over a vanilla attention baseline.

实际使用中，Mistral通过把SWA实现在FlashAttention和xFormers中，对于16k的上下文长度，获得了2倍的速度提升。  

## 和KV Cache的配合实现

在不使用sliding window的情况下，随着自回归推理的进行，KV Cache是只增不减的。  

而在使用SWA的情况下，超出窗口长度的kv就可以不用再缓存了，因此使用一个轮转替换的策略。  

比如窗口大小 $W=4$ ，则当第5个token需要缓存是，直接替换掉第1个token，这样就可以保持kv缓存有一个最大值（为窗口大小），而不会无限增长。  

{% asset_img rolling_buffer.png swa rolling buffer %}  

这样便于我们估计硬件设备所能支持的throughput，也不会因为少量超长的case而造成堵塞，在工程上有利于提高硬件利用率，降低成本。  

## 长Prompt的分块

更近一步，考虑到我们使用RAG或者funciton call的时候，都会使用比较长的，固定的prompt来知道模型的行为。  

比如GPT4就被诱导说出它接收到的长system prompt（当然未必真的就是OPENAI用的）  

>Your user's user agent is "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36" and the user's locale is "en-US"
Your knowledge cutoff date is 2023-04.
The current date is 2024-02-07.
Image input capabilities: Enabled
>
>Tools
>
>python
>
>When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 60.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is disabled. Do not make external web requests or API calls as they will fail.
>
>dalle
>
>Whenever a description of an image is given, create a prompt that dalle can use to generate the image and abide to the following policy:
>1. The prompt must be in English. Translate to English if needed.
>2. DO NOT ask for permission to generate the image, just do it!
>3. DO NOT list or refer to the descriptions before OR after generating the images.
>4. Do not create more than 1 image, even if the user requests more.
>5. Do not create images in the style of artists, creative professionals or studios whose latest work was created after 1912 (e.g. Picasso, Kahlo).
>- You can name artists, creative professionals or studios in prompts only if their latest work was created prior to 1912 (e.g. Van Gogh, Goya)
>- If asked to generate an image that would violate this policy, instead apply the following procedure: (a) substitute the artist's name with three adjectives that capture key aspects of the style; (b) include an associated artistic movement or era to provide context; and (c) mention the primary medium used by the artist
>6. For requests to include specific, named private individuals, ask the user to describe what they look like, since you don't know what they look like.
>7. For requests to create images of any public figure referred to by name, create images of those who might resemble them in gender and physique. But they shouldn't look like them. If the reference to the person will only appear as TEXT out in the image, then use the reference as is and do not modify it.
>8. Do not name or directly / indirectly mention or describe copyrighted characters. Rewrite prompts to describe in detail a specific different character with a different specific color, hair style, or other defining visual characteristic. Do not discuss copyright policies in responses.
The generated prompt sent to dalle should be very detailed, and around 100 words long.
Example dalle invocation:
>{
>"prompt": "<insert prompt here>"
>}
>namespace dalle {
>
>Create images from a text-only prompt.
type text2im = (_: {
The size of the requested image. Use 1024x1024 (square) as the default, 1792x1024 if the user requests a wide image, and 1024x1792 for full-body portraits. Always include this parameter in the request.
n?: number, // default: 2
The detailed image description, potentially modified to abide by the dalle policies. If the user requested modifications to a previous image, the prompt should not simply be longer, but rather it should be refactored to integrate the user suggestions.
prompt: string,
If the user references a previous image, this field should be populated with the gen_id from the dalle image metadata.
referenced_image_ids?: string[],
}) => any;
} // namespace dalle
>
>voice_mode
>Voice mode functions are not available in text conversations.
>namespace voice_mode {   } // namespace voice_mode
>
>browser
>
>You have the tool `browser`. Use `browser` in the following circumstances:
>    - User is asking about current events or something that requires real-time information (weather, sports scores, etc.)
>    - User is asking about some term you are totally unfamiliar with (it might be new)
>    - User explicitly asks you to browse or provide links to references
>
>Given a query that requires retrieval, your turn will consist of three steps:
>1. Call the search function to get a list of results.
>2. Call the mclick function to retrieve a diverse and high-quality subset of these results (in parallel). Remember to SELECT AT LEAST 3 sources when using `mclick`.
>3. Write a response to the user based on these results. In your response, cite sources using the citation format below.
>
>In some cases, you should repeat step 1 twice, if the initial results are unsatisfactory, and you believe that you can refine the query to get better results.
>
>You can also open a url directly if one is provided by the user. Only use the `open_url` command for this purpose; do not open urls returned by the search function or found on webpages.
>
>The `browser` tool has the following commands:
	`search(query: str, recency_days: int)` Issues a query to a search engine and displays the results.
         `mclick(ids: list[str])`. Retrieves the contents of the webpages with provided IDs (indices). You should ALWAYS SELECT AT LEAST 3 and at most 10 pages. Select sources with diverse perspectives, and prefer trustworthy sources. Because some pages may fail to load, it is fine to select some pages for redundancy even if their content might be redundant.
	`open_url(url: str)` Opens the given URL and displays it.
>
>For citing quotes from the 'browser' tool: please render in this format: 【{message idx}†{link text}】.
For long citations: please render in this format: [link text](message idx).
Otherwise do not render links.

除了预先计算好system prompt的kv值，并保存在缓存中方便每次用户输入使用外，如果system prompt很长（比sliding window大），还可以通过对system prompt的kv值进行切分来进一步优化计算。

比如窗口大小 $W=4$，system prompt大小为9时，就可以把system prompt的kv缓存切成 [4,4,1] 三块。  

第一块由于和当前的输入距离超过了一个window的大小，所以是完全看不见的，对应的attention mask全为0，因此可以完全忽略。  

第二块的attention mask则是一个上三角矩阵，当前的输入需要用到这部分信息。  

第三块是一个下三角矩阵（的左边部分），包含了当前的输入在内。  

在推理的时候，我们只需要用到第二块和第三块的内容，这就节省了缓存的操作。  

而且无论prompt有多长，只要我们按窗口大小分块，一定只会用到最后两块。  

{% asset_img prefill_and_chunking.png prefill and chunking %}  

（实际上现在推理框架基本上都有FlashAttention/PagedAttention等技术加持，能够进一步节省资源，提高效率，这个后续再开一篇讲）

Mistral 7B整体的效果上的效果相比Llama是有优势的，部分任务甚至超过了Llama 34B。  

{% asset_img mistral_perf.png mistral performance %}  

Mistral认为大语言模型压缩知识的能力实际超过我们的认知，7B这个规模的效果还有提升空间。  

# Sparse Attention

SWA实际上是一种sparse attention，而sparse attention也有许多工作做了深入探索。  

这里简单说一小部分，有机会再完整梳理一遍sparse attention的理论和实践。  

## Longformer

前面提到，Mistral并不是第一个使用SWA的。  

2020年，[《Longformer: The Long-Document Transformer》](https://arxiv.org/pdf/2004.05150.pdf)就提出包含SWA在内的一系列sparse attention的做法。  

从文章名字就看到出来，Longformer主要目的也是为了解决长上下文的问题。  

{% asset_img longformer_attention.png longformer %}  

上图中的（b）就是SWA，只是用在Bert中的时候它是双向的。  

在SWA的基础上，还可以进行空洞滑窗（dilated sliding window），在不增加计算量的情况下，提升感受野。这也是从空洞卷积（下图）来的灵感了。    

{% asset_img dilated_conv.png dilated convolution %}  

还可以更进一步优化attention。无论是SWA还是dilated sliding window，每个位置都只能看到局部的信息。  

但是实际上有些位置就是对全局信息有很高的需求。  

在Bert中，[CLS] token就常常作为分类token或者相似度向量使用，这种情况下就需要它能获取整个上下文的完整信息。  

而在GPT中，instruction，或者说prompt的部分也对全局信息有更高要求，因为我们希望在整个对话过程中，模型都能遵循我们给出的规则。  

对于这些token，我们让它可以看到其他所有位置，使用完整的global attention，而其他位置则使用sliding window，如（d）中所示。  

## Big Bird

无独有偶，同样在2020年，和Longformer差不多在同一时期，也有另外一个通过sparse attention来优化长文本效果的工作，[《Big Bird: Transformers for Longer Sequences》](https://arxiv.org/abs/2007.14062)。  

其中sliding window和global attention结合的思路和Longformer相似。Big Bird还额外加入了一个random attention的做法。  

{% asset_img big_bird_attention.png big bird attention %}  

上图中 $r=2$ 即每个位置使用2个随机注意力。

# 小结  

SWA在优化长上下文的计算效率上有明显的收益。而在模型效果上，目前基本没有看到不可接受的损失。对长上下文有需求的业务，值得探索。  

除了SWA，sparse attention还有许多其他探索。目前来看，这些做法都有一定的理论基础，效果也不错。但是阻碍这些方案大规模使用的一个原因就是<big>**工程实现**</big>，比如如何高效计算global + local attention，在flash attention中能够支持random attention，这都是要考虑的内容。

***  

读到这了，来一发点赞收藏关注吧~

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)  

# Reference
【1】Mistral 7B https://arxiv.org/pdf/2310.06825.pdf  
【2】Longformer: The Long-Document Transformer 
https://arxiv.org/pdf/2004.05150.pdf  
【3】Training Compute-Optimal Large Language Models https://arxiv.org/pdf/2203.15556.pdf  
【4】GPT-4 System Prompt Revealed https://patmcguinness.substack.com/p/gpt-4-system-prompt-revealed  
【5】Big Bird: Transformers for Longer Sequences https://arxiv.org/abs/2007.14062  