
import requests
from bs4 import BeautifulSoup as bs
from concurrent import futures
sum_num = 0
## label = [' adult ', ' child ', ' person ', ' people ', ' girl ', ' boy ', ' woman ', ' man ', ' kid ' ' elderly ', ' teenager ', ' baby ']
def get_img_urls_download(page_num):
    key = 'father'
    try:
        global sum_num
        url = f'https://unsplash.com/napi/search/photos?query={key}&xp=&per_page=20&page='+str(page_num)
#         headers = {
# #                     "authority":"unsplash.com",
#                     "referer":"https://unsplash.com/search/photos/mobile-phone",
#                     "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36"
#                     }
        raw_data =  requests.get(url).json()
        link_list = raw_data.get("results")
        print("downloading %d page"%page_num)
        for link in link_list:
            link = link.get("links").get("download")
#             print(link)
            img = requests.get(link).content
#             print(img)
            with open(f"./unleash/{key}_%d.jpg"%sum_num, "wb") as f:
                f.write(img)
            sum_num +=1
            
        print("page %d finished"%page_num)
    except Exception:
        print("page %d failed"%page_num)
        
def download_start(end_page):
    workers = 100
    with futures.ThreadPoolExecutor(workers) as e:
        e.map(get_img_urls_download, [page_num for page_num in range(end_page)])
        
download_start(1000) 
