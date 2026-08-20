import os, json, datetime, time, multiprocessing.pool, random
from langchain.docstore.document import Document
from langchain_core.messages.ai import AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import docx2txt
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import CBextension

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain

import panel as pn
import param

folder = 'vdb'
flst = [f.lower() for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

doc_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = []
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
struct_docs = []
for file in flst:
    pth = os.path.join(folder, file)
    surfix = pth.split('.')[-1]
    if surfix in ('docx', 'doc'):
        document = docx2txt.process(pth)
        docs += [Document(page_content=document, metadata={'source': file})]
    elif surfix in {'pdf'}:
        loader = PyPDFLoader(pth)
        docs += loader.load()
    elif surfix in ('txt', 'csv'):
        loader = TextLoader(pth, encoding='utf8')
        docs += loader.load()
    elif surfix in ('htm','html'):
        html = open(pth, encoding = 'utf-8').read()
        soup = BeautifulSoup(html, features="html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        docs += [Document(page_content=soup.get_text().replace('\n\n',''), metadata={"source": file})]
    elif surfix in {'json'}:
        with open(pth, 'r', encoding = 'UTF8') as f:
            doc = json.load(f)
        struct_docs += [Document(page_content=str(_), metadata={"source": file}) for _ in doc]
    elif surfix in ('md', 'markdown'):
        # Read the markdown file
        with open(pth, 'r', encoding='utf-8') as f:
            md_doc = f.read()
        struct_docs += markdown_splitter.split_text(md_doc)

docs = doc_splitter.split_documents(docs) + struct_docs

api_keys = os.environ["API_KEYS"].split(",")
os.environ["GOOGLE_API_KEY"] = api_keys[0]
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # models/text-embedding-004
db = Chroma.from_documents(docs, embeddings) #, persist_directory="db")
db_threshold = 0.35
num_docs = 10
class shift_llms():
    resources = [("ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.5, google_api_key = api_key)",
                 os.environ["API_KEYS"].split(","))]
    def __init__(self):
        for resource in self.resources:
            random.shuffle(resource[1])
        print(self.resources)

    def invoke(self, query):
        tag = True
        nllms = len(self.resources)
        while tag and nllms > 0:
            resource = self.resources[0]
            print(self.resources)
            keys = resource[1]
            cnt = len(keys)
            while tag and cnt > 0:
                api_key = keys[0]
                try:
                    llm = eval(resource[0])
                    with multiprocessing.pool.ThreadPool() as pool:
                        rep = pool.apply_async(llm.invoke, args = (query,)).get(timeout=15)
                except multiprocessing.TimeoutError:
                    print(f'key shift {resource[0]}')
                    # push invalid reource to end of queue
                    self.resources[0][1].append(self.resources[0][1].pop(0))
                    cnt -= 1
                    continue
                tag = False
            if tag: print(f'model {resource[0]} exhaused')
            # push invalid reource to end of queue
            self.resources.append(self.resources.pop(0))
            nllms -= 1
            print(self.resources)
        return rep if not tag else AIMessage('Resources Exhaused')
llm = shift_llms()
# RAG parameters
# chain_type = 'stuff'
template_calls = """
用编程技能结合Context回答最后一个Question，如果可以用Context里的function calls回答，就按照Example的格式回复纯文本，
如果不能就回复["{'value':'<原因>', 'description':'无法回复的原因'}"]，<>表示可替换信息，如果Question里缺少所需信息，一定要在原因里说明。
Example是一个list，里面每个元素都是独立的一行Python代码，可以自行发挥但不能超过一行，别把代码放print()里。当前时间在context里，可以自行推测所需时间信息。

Example:
["CBextension.DishSalesForecaster.sales_prediction(CBextension.get_date(0)['value'], '2021-02-01 00:00:00.000000')", 
"CBextension.DishSalesForecaster.sales_prediction('2020-02-01 00:00:00.000000', CBextension.get_date(30)['value'], dishes = ['jelly', 'red bean'])",
"CBextension.docdb_search({'$and':[{'ip_location':'美国'},{'fans':{'$gt':800}}]}, 'creators', '美国粉丝数大于800的头10位网红', [('fans', -1)], 10)",
"CBextension.docdb_search({"$or":[{"Chinese_name": {"$regex": "红豆"}},{"Chinese_discription": {"$regex": "红豆"}}]}, 'Menu', '翻译红豆产品')"]

Context:
当前时间%Y-%m-%d %H:%M:%S.%f：{now}。
{call: CBextension.get_date(n),
description: 获取n天后的日期，负数n获取n天前的日期,
return: a dictionary of {value: str, description: str},
parameters: {required: {n: number of days}}}
{call: CBextension.DishSalesForecaster.sales_prediction(start_date, end_date),
description: 预测给定时间范围内的销售额,
return: a dictionary of {value: str, description: str},
parameters: {required: {start_date: start date string %Y-%m-%d %H:%M:%S.%f, end_date: end date string %Y-%m-%d %H:%M:%S.%f,
optional:{dishes: 产品英文名，可以模糊匹配}}}}
{call: CBextension.docdb_search(query, collection, desc, order, top),
description: 搜索Mongodb,
return: a dictionary of {value: json str, description: str},
parameters: {required: {query: search criteria, collection: the collection to be searched, desc: description of the search},
optional: {order: sorting, top: limit}}
collection_data: {
'creators':{"user_id": "Unique identifier for each creator.",
  "nickname": "The display name of the creator.",
  "avatar": "URL to the creator's profile image.",
  "profile_intro": "Enriched description of the creator. This text combines the original profile description with appended tag-based insights (e.g., zodiac, age, profession, college).",
  "ip_location": "Location inferred from the creator’s IP address.",
  "follows": "Number of accounts the creator is following.",
  "fans": "Number of followers the creator has.",
  "interaction": "Overall engagement metric on the creator’s profile.",
  "last_modify_ts": "Time when the creator’s profile or content was last updated (displayed in datetime format).",
  "pic_per_normal_note": "Average number of pictures per normal (non-video) post.",
  "video_ratio": "Ratio of video posts to total posts.",
  "hot_note_count": "Count of \"hot\" (popular) posts by the creator.",
  "total_share_counts_hot_ratio": "Ratio of total share counts relative to hot posts, reflecting engagement on popular content.",
  "last_note2now": "number of days elapsed from the creator's most recent post to the current time.",
  "last_hot_note2now": "number of days elapsed from the creator's most recent hot post to the current time.",
  "weighted_total_share_counts": "Weighted sum of share counts across posts, emphasizing certain posts over others.",
  "liked_count": "Total number of likes received across all posts.",
  "collected_count": "Total number of times posts were collected (saved or bookmarked).",
  "comment_count": "Total number of comments received across all posts.",
  "share_count": "Total number of shares accumulated from all posts.",
  "note_count": "Total count of posts (notes) made by the creator.",
  "location": "three Standardized location categories derived from tag information (TX, US, abroad).",
  "ff_ratio": "Ratio of fans to follows, indicating the creator's influence.",
  "age_koc": "Account age in days, calculated as the difference between the newest and oldest posts.",
  "is_female": "Binary indicator for gender; 1 indicates female, 0 indicates male.",
  "min": "The minimum (earliest) post timestamp for the creator.",
  "max": "The maximum (latest) post timestamp for the creator.",
  "post_span": "Time span between the earliest and latest posts.",
  "first_post_time": "Timestamp of the creator's first post.",
  "account_length": "Duration in days from the creator's first post to the last modification timestamp.",
  "history_avg": "Average time interval between consecutive posts.",
  "history_std": "Standard deviation of the time intervals between posts.",
  "post_avg": "Average posting frequency (e.g., average number of posts per day).",
  "post_std": "Standard deviation of the posting frequency.",
  "liked_90": "Sum of likes received on posts made within the last 90 days (relative to the creator's last modification timestamp).",
  "collected_90": "Sum of collected counts from posts in the 90-day window.",
  "comment_90": "Sum of comments from posts in the last 90 days.",
  "share_90": "Sum of shares from posts within the 90-day period.",
  "note_count_90": "Count of posts made within the last 90 days.",
  "liked_180": "Sum of likes received on posts over the past 180 days.",
  "collected_180": "Sum of collected counts from posts in the 180-day window.",
  "comment_180": "Sum of comments received on posts over the past 180 days.",
  "share_180": "Sum of shares from posts within the 180-day period.",
  "note_count_180": "Count of posts made within the last 180 days.",
  "图文报价(RMB)": "新红网站所给出的creator评估价格，单位为人民币",
  "图文报价有对号": "该名creator是否进行过商业合作，1代表是，0代表否",
  "平台等级": "小红书对于creator评估的平台等级从最低1到最高10，等级9和等级10被归类为等级9，未知等级被推测为等级3",
  "活跃粉丝占比(仅>1K)": "新红网站调查得到的creator粉丝中活跃粉丝的占比，单位为%，如果新红网站未提供，则标为null",
  "粉丝女性比例": "新红网站调查得到的creator粉丝中女性粉丝的占比，单位为%，如果新红网站未提供，则标为null",
  "粉丝年龄<18": "新红网站调查得到的creator粉丝中小于18岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null",
  "粉丝年龄18-24": "新红网站调查得到的creator粉丝中大于18岁且小于24岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null",
  "粉丝年龄25-34": "新红网站调查得到的creator粉丝中大于25岁且小于34岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null",
  "粉丝年龄35-44": "新红网站调查得到的creator粉丝中大于35岁且小于44岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null",
  "粉丝年龄>44": "新红网站调查得到的creator粉丝中大于44岁的粉丝的占比，单位为%，如果新红网站未提供，则标为null",
  "兴趣标签": "新红网站调查得到的creator发帖倾向标签的前五名，每行记录一个标签的排名、标签内容和占比",
  "地域分布": "新红网站调查得到的creator粉丝地理分布的前五名，每行记录一个地域的排名、地域内容和占比",
  "score1_account_influence": {
    "description": "综合衡量创作者在平台上的互动表现、受众反馈质量以及持续活跃度等多个维度的账号影响力指标。该评分通过整合三个关键子维度（互动效能、互动质量、账号活跃度），旨在识别在真实用户中具有高影响力、强曝光互动能力的创作者。每个子评分均通过 Logistic 映射函数进行标准化，避免极端值并增强中间段区分度。",
    "score1a_efficiency": "基于点赞、收藏、评论三种核心互动行为，并引入时间衰减机制，衡量创作者在内容层面的真实互动效能。该评分更关注近期内容的热度，强调时效性与真实互动表现。对点赞、收藏、评论数引入时间衰减（指数函数），再分别加权（35%、45%、20%），最后使用 Logistic 映射函数标准化并缩放至 0–50 范围。",
    "score1b_quality": "通过“互动渗透率”（总互动数 / 粉丝数）与“评论占比”或“活跃粉丝占比”两个维度，衡量互动的真实价值与深度，避免高互动但低质量反馈的账号获得高分。对渗透率与评论占比/活跃粉丝比使用 Logistic 函数进行映射后加权（70%、30%），并映射到 0–30 分的范围。",
    "score1c_activity": "评估创作者发帖的规律性与持续性。考虑三个维度：平均发帖间隔（越频繁越好）、发帖时间标准差（越稳定越好）、近 90 天发帖数量（反映近期活跃程度）。对间隔与波动性使用反向 Logistic 映射（更稳定得分越高），发帖频率则线性映射为满分 12。整体得分加权后映射至 0–20。"
  },
  "score2_content_media": {
    "description": "综合衡量创作者在平台上的内容传播力指标。该评分通过整合三个关键子维度（传播效能、爆款质量、视觉表达度），旨在识别在真实用户中具有高影响力、强曝光互动能力的创作者。每个子评分均通过 sigmoid 映射函数进行标准化，避免极端值并增强中间段区分度。",
    "score2a_media": "基于该用户帖子的分享互动行为，并引入时间衰减机制，衡量创作者在内容层面的真实互动效能。该评分更关注近期内容的传播力，强调时效性与真实互动表现。对分享数引入时间衰减（指数函数），快速期半衰期25天（每日衰减2%），长尾期半衰期175天（每周衰减2%），基于全量180天内分享总数据95%分位数动态设定，最后使用 sigmoid 映射函数标准化并缩放至 0–50 范围。",
    "score2b_hot": "根据该用户粉丝数划分为千粉级与粉丝数不到一千的素人级，并将其中总互动数高于一定阙值的帖子定义为爆款帖子，粉丝数越少阙值越低。对用户的爆款笔记数除以基于全量180天内分享总数据95%分位数动态设定并给予70%的权重，对用户的爆款分享占比除以基于全量180天内分享总数据95%分位数动态设定并给予30%的权重，将两者权重相加后的结果使用 sigmoid 映射函数标准化并缩放至 0–30 范围。",
    "score2c_picture": "对用户的图片密度（非视频笔记图片总数/非视频笔记数，上限8张）除以基于全量180天内分享总数据95%分位数动态设定并给予60%的权重，对用户的视频笔记占比除以基于全量180天内分享总数据95%分位数动态设定并给予40%的权重，将两者权重相加后的结果使用 sigmoid 映射函数标准化并缩放至 0–20 范围。"
  },
  "score3_content_quality": {
    "description": "旨在从多个维度综合衡量创作者内容的专业性、独特性、感染力和相关性。该指标通过整合四个关键维度的评分，为品牌识别优质内容创作者提供定量依据，确保投放资源能获得最佳营销效益。每个维度的得分都经过了sigmoid函数的分布校正",
    "score3a_originality": "基于同自身内容(内部)和其他创作者内容(外部)比较后, 旨在定量评估创作者产出差异化内容的能力",
    "score3b_vertical": "通过与既定的目标领域集合对比, 定量分析创作者的内容专注度，有助于发现在目标领域持续有专业深度贡献的创作者",
    "score3c_sentiment": "通过NLP技术量化笔记中的情感特征，旨在识别那些能够传递丰富、真实情感体验，从而与受众产生深度共鸣的内容创作者",
    "score3d_keyword": "通过模糊文本匹配技术，识别那些持续产出与目标品牌/产品高度相关内容的创作者，确保推荐的KOC真正聚焦于特定主题"
  },
  "score4_target_audience_match": {
    "description": "综合评估创作者粉丝群体与目标客群在兴趣标签、地域分布、年龄结构、性别比例四个维度的匹配度。通过层次化标签模型、梯度衰减模型、向量相似度匹配、非线性补偿函数等算法，量化内容创作者对目标受众的精准程度。",
    "score4a_tag": "兴趣标签匹配度，衡量粉丝兴趣标签（美食/生活记录/探店）与目标客户群的契合程度。采用双层评估体系，既考核标签覆盖率阈值达标度，也评估匹配精准度。计算方法为重合率得分 = (粉丝兴趣标签总占比/25%) × 50",
    "score4b_region": "地域聚焦度，通过海外用户占比阶梯得分与DFW区域地理关键词覆盖度复合计算。重点识别具有本土化运营价值的创作者。计算方法为海外比例得分 = 逆向阶梯模型（≥50%:10分，每降10%扣2分）+地理覆盖得分 = 10 × (地理关键词笔记覆盖比例 / 基准值)",
    "score4c_age": "年龄分布匹配度，通过粉丝年龄分布向量与目标向量的余弦相似度$`\theta_0`$ = (18<: 10%, 18-24岁:30% + 25-34:30% + 35-44: 15% + >44: 15%) 计算。识别年龄结构吻合目标消费决策群体的账号。计算方法为得分 = 20 × [1 + 余弦相似度(θ,θ₀)] / 2",
    "score4d_gender": "性别匹配度，采用拉普拉斯衰减函数对女性比例偏差进行双向惩罚。当粉丝女性比例偏离80%基准时，得分非线性衰减。计算方法为得分 = 10 × exp(-2×|P_女粉-80%|/15)"
  },
  "score5_business_coop": {
    "description": "综合衡量创作者在平台上的商业合作潜力指标。该评分通过整合三个关键子维度（报价合理性、平台信用等级、合作意向信号），旨在识别在真实用户中具有合作潜力的创作者。每个子评分均通过 sigmoid 映射函数进行标准化，避免极端值并增强中间段区分度。计算此维度分数的数据来源于新红网站。",
    "score5a_price": "基于新红网站提供的报价评估引入衰减机制，使100元以上的报价越高，得分越少。",
    "score5b_Credit_Level": "基于小红书平台提供的九个用户信用等级转化为分数，低于三级的被视为三级，高于八级的得到略低于八级的分数。",
    "score5c_coop_history": "有过合作历史的用户得10分，没有则不得分。"
  },
  "total_score":"将五个维度加权相加后的总分，代表了该用户作为KOC总体的合作宣传能力。"},
'contents':{"note_id": "Unique identifier for each post.",
  "user_id": "Identifier linking the post to its creator.",
  "title": "Title of the post.",
  "note_body": "Description or body text of the post.",
  "tag_list": "JSON-formatted string containing additional tags or metadata associated with the post.",
  "image_count": "Count of image URLs associated with the post.",
  "content_type_video": "Indicator or label specifying whether the post is a video post. This field differentiates video content from other types. 1 means it is video post and 0 means it is normal post.",
  "hot_note": "Indicator or metric denoting whether the post is considered popular or \"hot\". 1 means it is hot and 0 means it is not.",
  "post_time": "Time representing when the post was created.",
  "last_update_time": "Time indicating the last time the post was updated.",
  "scraped_time": "Time used for content extraction or indicating when the data was processed.",
  "elapsed_time": "Time between the content was posted and scraped.",
  "liked_count": "Number of likes the post has received.",
  "collected_count": "Number of times the post has been saved or bookmarked.",
  "comment_count": "Number of comments on the post.",
  "share_count": "Number of shares the post has received.",
  "interaction_count": "A computed metric representing the total interactions (e.g., sum of likes, collections, and shares) for the post."},
"Menu": {"English_name": "英文名字",
      "English_discription": "英文描述",
      "Chinese_name": "中文名字",
      "Chinese_discription": "中文描述",
      "spanish_name": "西班牙语名字",
      "Spanish_discription": "西班牙语描述"},
"Top_ingredients_introduction": {"Ingredients": "配方",
      "Ingredient_English": "配方英文",
      "Ingredient_chinese": "配方中文",
      "Ingredient_Spanish": "配方西班牙语"},
"Promotion": {"Month": "月份英文名",
      "Anchor_Dish": "主打产品",
      "Rotating_Feature_Dish": "轮转产品",
      "Seasonal_Note": "季度特色",
      "Pricing_Idea": "价格推荐"}}}

{QA_history}

Question: {question}
"""

template = """你是美国达拉斯一家鲜芋仙店铺的店长助理，请结合以下Context和市场公关知识回答最后一个Question.
回复采用Markdown格式。‘Analysis:’里的内容都是你的分析助手提供的，不用怀疑，优先用这里的信息回答，除非这里的信息提示没有相关功能。
如果‘|analysis retults’为无法回复，会提供原因，若原因为没有相关功能，可以适当自由发挥，其他原因请转达提供的信息不要自由发挥。

Context:
当前时间%Y-%m-%d %H:%M:%S.%f：{now}。
{context}

{QA_history}

Question: {question}
Answer:
"""

class rag_ext:
    chat_history = []
    llm = None
    db = None
    call_prompt_template = None
    prompt_template = ''
    history_length = 0
    doc_num = 0
    def __init__(self, llm, db, prompt_template, **kwargs):
        self.llm = llm
        self.db = db
        self.call_prompt_template = kwargs.pop("prompt_template_call", None)
        self.prompt_template = prompt_template
        self.history_length = kwargs.pop('history_length', 50)
        self.doc_num = kwargs.pop('doc_num', 10)
        self.params = kwargs # kwargs is dictionary
    def query(self, query):
        prompt = self.prompt_template.replace('{question}', query)
        current_time = datetime.datetime.now()
        prompt = prompt.replace('{now}', str(current_time))
        self.chat_history = self.chat_history[-self.history_length:]
        qa_history = '\n\n'.join('Question:' + q + '\nAnalysis : ' + x[:100] + '\nAnswer: ' + a 
                                 for q, x, a in self.chat_history) if len(self.chat_history) > 0 else ''
        prompt = prompt.replace('{QA_history}', qa_history)
        ref_docs = self.db.similarity_search_with_score(query, k = self.doc_num)
        ref_docs.sort(key = lambda x: x[1], reverse=False)
        ref_docs = [{'doc': _[0], 'score': _[1]} for _ in ref_docs \
                    if self.params.get('doc_score_lower', float('-inf')) < _[1] < self.params.get('doc_score_upper', float('inf'))]
        context = '\n\n'.join(_['doc'].page_content for _ in ref_docs) if len(ref_docs) > 0 else 'No reference is given'
        prompt = prompt.replace('{context}', context)
        prompt += '\nAnalysis: '
        if self.call_prompt_template:
            call_prompt = self.call_prompt_template.replace('{question}', query)
            call_prompt = call_prompt.replace('{now}', str(current_time))
            call_prompt = call_prompt.replace('{QA_history}', qa_history)
            answer = self.llm.invoke(call_prompt)
            content = ''
            if len(answer.content) > 5:
                results = ''
                try:
                    content = answer.content
                    content = content[content.find('['):content.rfind(']')+1]
                    calls = json.loads(content)
                    for call in calls:
                        call = eval(call)
                        results += '|' + call['description'] + ':' + str(call['value'])
                except:
                    results += '|analysis break:' + answer.content + '|'+ content
            else:
                results = '|analysis break:' + answer.content
            prompt += results
            time.sleep(1)
        else:
            results = None
        prompt += '\nAnswer: '
        answer = self.llm.invoke(prompt)
        self.chat_history.append((query, results, answer.content))
        return {'answer': answer.content, 'source_documents': ref_docs, 'generated_question': prompt+'\n'+content}

qa = rag_ext(llm=llm, db=db, prompt_template = template
         , doc_num = num_docs, doc_score_upper = db_threshold, history_length = 10, prompt_template_call = template_calls)

pn.config.loading_spinner = 'petal'
pn.config.loading_color = 'black'
pn.extension()
class conv_rag():
    panels = []
    def convchain(self, query):
        self.panels = self.panels[-15:]
        if not query:
            return pn.WidgetBox(pn.Row(pn.layout.HSpacer(), pn.pane.Markdown("聊天历史……", styles={'color': '#cceeff'}), pn.layout.HSpacer()
                                       , sizing_mode="scale_width"), scroll=True)
        result = qa.query(query)
        db_response = [doc['doc'].metadata['source'] + ' ' + str(doc['score']) for doc in result["source_documents"]]
        db_query = result["generated_question"] + ('\nReferences:\n' + '\n'.join(doc for doc in db_response) if len(db_response)>0 else "No Reference")
        answer = result['answer'] # + '\nReferences:\n' + '\n'.join(doc for doc in db_response) if len(db_response)>0 else "I don't know"
        self.panels.extend([
            # pn.Row('Prompt:', pn.pane.Str(db_query, sizing_mode="scale_width", styles={'background-color': '#cce6ff'})),
            pn.Row('小助理：', pn.pane.Markdown(answer, sizing_mode="scale_width", 
                                            styles={'background-color': '#cce6ff', 'border-radius': '10px', 'padding': "15px"})),
            pn.Row('用户：', pn.pane.Markdown(query, sizing_mode="scale_width"))
        ])
        inp.value = ''  #clears loading indicator when cleared
     
        return pn.WidgetBox(*reversed(self.panels), scroll=True, sizing_mode="fixed", width=1000)

cb = conv_rag()
inp = pn.widgets.TextInput(placeholder='请输入您的要求……', sizing_mode="fixed", width=800, align="center")
conversation = pn.bind(cb.convchain, inp) 
tab1 = pn.Column(
    pn.Row(pn.layout.HSpacer(), inp, pn.layout.HSpacer(), sizing_mode="scale_width"),
    pn.layout.Divider(),
    pn.panel(conversation, loading_indicator=True),
    pn.layout.Divider()
)

dashboard = pn.Row(pn.layout.HSpacer(), pn.Column(
    pn.Row(pn.pane.Markdown('# 鲜芋仙小助理')),
    pn.Tabs(('会话',tab1), sizing_mode="fixed", width=1000),
    sizing_mode="fixed", width=1024, align="center"
), pn.layout.HSpacer())
server = pn.serve(dashboard,title="AI Assistant", port = 8002)