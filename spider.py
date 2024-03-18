import os
import requests
import re
import urllib

#from bs4 import BeautifulSoup

def is_file(url):
    """判断一个链接是否是文件"""
    return not url.endswith('/')

def download(url):
    """下载文件并按原路径格式保存到本地"""
    full_name = url.split('//')[-1]
    filename = full_name.split('/')[-1]
    dirname = "/".join(full_name.split('/')[:-1])

    if os.path.exists(dirname):
        pass
    else:
        os.makedirs(dirname, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, full_name)
    except Exception as e:
        print("exception catched............")
        print(e)

def get_url(base_url):
    """获取给定网址中的所有链接"""
    text = requests.get(base_url).text
    #reg = '<a href=" ">.*</a >'
    reg = '<a href=".*">(.*)</a>'

    urls = [base_url + url for url in re.findall(reg, text) if ((url != '/wp-content/') and (url != '?C=N;O=D') and (url != '?C=M;O=A') and (url != '?C=S;O=A') and (url != '?C=M;O=A'))]

    return urls

def get_file(url):
    """递归下载网站的文件"""
    if is_file(url):
        print("下载："+ url + "\n\r")
        try:
            download(url)
        except:
            pass
    else:
        urls = get_url(url)
        for u in urls:
            get_file(u)

if __name__ == '__main__':
    # 以维基泄密网站为例
    #get_file('https://medicatechusa.com/wp-content/uploads/')
    get_file('https://medicatechusa.com/wp-content/uploads/2024/02/')
