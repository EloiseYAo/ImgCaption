# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import requests
import random
import json
from hashlib import md5
from models.test_input import Input
from models.inference import Inference

import matplotlib.pyplot as plt

# Set your own appid/appkey.
appid = '20210423000796816'
appkey = 'j4etogc54_tLKt6ycaSW'

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'en'
to_lang =  'zh'

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path

'''test_input'''
# inference = Input()
# img, en_raw_caption = inference.get_caption()
# en_raw_caption=" ".join(en_raw_caption)

'''inference'''
inference = Inference(1)
img, en_raw_caption = inference.get_caption()
# en_raw_caption=" ".join(en_raw_caption)

en_raw_caption=inference.get_true_caption()[0][8:]

print(en_raw_caption)
en_raw_caption=en_raw_caption[:-5]
print(en_raw_caption)
en_caption = en_raw_caption
if en_raw_caption.find('<UNK>') != -1:
    en_caption=en_raw_caption.replace('<UNK>','【unknow】')
print(en_caption)

query = en_caption

# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

salt = random.randint(32768, 65536)
sign = make_md5(appid + query + str(salt) + appkey)

# Build request
headers = {'Content-Type': 'application/x-www-form-urlencoded'}
payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

# Send request
r = requests.post(url, params=payload, headers=headers)
result = r.json()

# Show response
print(json.dumps(result, indent=4, ensure_ascii=False))

zh_caption=result['trans_result'][0]['dst']
print(zh_caption)



# plt.rc("font",family='MicroSoft YaHei',weight="bold")

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.imshow(img[0])
plt.title(zh_caption,fontsize=14)
print()
plt.show()