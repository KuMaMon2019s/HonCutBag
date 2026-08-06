#!/usr/bin/env python3
"""
IP 黑名单过滤模块

过滤提示词中的品牌名、版权角色、真实人名，避免 IP 侵权风险。
"""

import re
from typing import List, Tuple


# 品牌名黑名单
BRAND_BLACKLIST = [
    # 运动品牌
    "Nike", "耐克", "Adidas", "阿迪达斯", "Puma", "彪马",
    "Under Armour", "安德玛", "New Balance", "新百伦",
    # 奢侈品牌
    "Gucci", "古驰", "Louis Vuitton", "LV", "香奈儿", "Chanel",
    "Dior", "迪奥", "Prada", "普拉达", "Hermes", "爱马仕",
    # 科技品牌
    "Apple", "苹果", "iPhone", "iPad", "MacBook",
    "Samsung", "三星", "Huawei", "华为", "Xiaomi", "小米",
    # 餐饮品牌
    "Coca-Cola", "可口可乐", "Pepsi", "百事可乐",
    "McDonald's", "麦当劳", "KFC", "肯德基", "Starbucks", "星巴克",
    # 汽车品牌
    "BMW", "宝马", "Mercedes-Benz", "奔驰", "Audi", "奥迪",
    "Tesla", "特斯拉", "Toyota", "丰田", "Honda", "本田",
    # 其他知名品牌
    "Disney", "迪士尼", "Marvel", "漫威", "DC Comics",
    "Coca-Cola", "可口可乐", "Pepsi", "百事可乐",
]

# 版权角色黑名单
CHARACTER_BLACKLIST = [
    # 漫威/DC 角色
    "Iron Man", "钢铁侠", "Spider-Man", "蜘蛛侠", "Batman", "蝙蝠侠",
    "Superman", "超人", "Captain America", "美国队长", "Thor", "雷神",
    "Hulk", "绿巨人", "Black Widow", "黑寡妇", "Wonder Woman", "神奇女侠",
    "Joker", "小丑", "Harley Quinn", "哈莉·奎茵",
    # 迪士尼/皮克斯角色
    "Mickey Mouse", "米老鼠", "Donald Duck", "唐老鸭",
    "Elsa", "艾莎", "Anna", "安娜", "Olaf", "雪宝",
    "Woody", "胡迪", "Buzz Lightyear", "巴斯光年",
    "Simba", "辛巴", "Mufasa", "木法沙",
    # 日本动漫角色
    "Goku", "孙悟空", "Naruto", "鸣人", "Luffy", "路飞",
    "Pikachu", "皮卡丘", "Doraemon", "哆啦A梦",
    "Sailor Moon", "美少女战士", "Gundam", "高达",
    # 其他知名角色
    "Harry Potter", "哈利波特", "Hermione", "赫敏", "Ron Weasley", "罗恩",
    "Frodo", "佛罗多", "Gandalf", "甘道夫",
    "James Bond", "007", "Indiana Jones", "夺宝奇兵",
]

# 真实人名黑名单
PERSON_BLACKLIST = [
    # 导演/电影人
    "张艺谋", "陈凯歌", "冯小刚", "姜文", "王家卫",
    "斯皮尔伯格", "Spielberg", "诺兰", "Nolan", "卡梅隆", "Cameron",
    "宫崎骏", "新海诚", "昆汀", "Tarantino",
    # 明星/名人
    "成龙", "李连杰", "周杰伦", "刘德华", "梁朝伟",
    "Taylor Swift", "Beyoncé", "Kanye West", "Elon Musk",
    # 政治家
    "Obama", "奥巴马", "Trump", "特朗普", "Biden", "拜登",
    "Putin", "普京", "Merkel", "默克尔",
    # 历史人物
    "爱因斯坦", "Einstein", "牛顿", "Newton", "达芬奇", "Da Vinci",
    "马克思", "Marx", "列宁", "Lenin",
]


def sanitize_prompt(prompt: str) -> Tuple[str, List[str]]:
    """
    过滤提示词中的 IP 风险内容
    
    Args:
        prompt: 原始提示词
    
    Returns:
        (过滤后的提示词, 被过滤的关键词列表)
    """
    filtered_terms = []
    sanitized = prompt
    
    # 合并所有黑名单
    all_blacklist = BRAND_BLACKLIST + CHARACTER_BLACKLIST + PERSON_BLACKLIST
    
    for term in all_blacklist:
        # 使用正则表达式进行大小写不敏感的匹配
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(sanitized):
            filtered_terms.append(term)
            # 替换为 [FILTERED]
            sanitized = pattern.sub("[FILTERED]", sanitized)
    
    return sanitized, filtered_terms


def check_prompt_risk(prompt: str) -> dict:
    """
    检查提示词的 IP 风险等级
    
    Args:
        prompt: 提示词
    
    Returns:
        {
            "risk_level": "low" | "medium" | "high",
            "filtered_terms": [...],
            "sanitized_prompt": "..."
        }
    """
    sanitized, filtered_terms = sanitize_prompt(prompt)
    
    # 根据过滤词数量判断风险等级
    if len(filtered_terms) == 0:
        risk_level = "low"
    elif len(filtered_terms) <= 2:
        risk_level = "medium"
    else:
        risk_level = "high"
    
    return {
        "risk_level": risk_level,
        "filtered_terms": filtered_terms,
        "sanitized_prompt": sanitized,
        "original_prompt": prompt
    }


if __name__ == "__main__":
    # 测试
    test_prompts = [
        "A man wearing Nike shoes walks down the street",
        "钢铁侠和蜘蛛侠在战斗",
        "张艺谋风格的电影画面",
        "一个普通的城市街景",
    ]
    
    for prompt in test_prompts:
        result = check_prompt_risk(prompt)
        print(f"原始: {prompt}")
        print(f"风险: {result['risk_level']}")
        print(f"过滤: {result['filtered_terms']}")
        print(f"结果: {result['sanitized_prompt']}")
        print("-" * 60)
