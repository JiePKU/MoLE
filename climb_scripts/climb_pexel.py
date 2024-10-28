
import requests
import json
import os
import time
import pprint


path = "./pexel_hand/"

if not os.path.exists(path):
    os.makedirs(path)

# label = ['adult', 'child', 'person', 'people', 'girl', 'boy', 'woman', 'man', 'kid' 'elderly', 'teenager', 'baby', 'mother', 'father']

label = ['hand']
def get_url_down(page):
    for key_word in label:
        base_url = f'https://www.pexels.com/en-us/api/v3/search/photos?page={page}&per_page=24&query={key_word}&orientation=all&size=all&color=all&seo_tags=true'
        # base_url = f'https://www.pexels.com/en-us/api/v3/search/photos?page={page}&per_page=24&query={key_word}&orientation=all&size=all&color=all&seo_tags=true'
        url, payload_string = base_url.split("?")
        payload = {word.split("=")[0]:word.split("=")[1] for word in payload_string.split("&")}
        
        headers = {"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) " 
                                "Chrome/106.0.0.0 Safari/537.36", "secret-key": "H2jk9uKnhRmL6WPwh89zBezWvr"}

        session = requests.Session()
        response = session.get(url)
        cookies = session.cookies.get_dict()
        resp = requests.get(url, headers=headers, cookies=cookies, params=payload)
        json_dict = json.loads(resp.text)
        # pprint.pprint(json_dict)

        data_list = json_dict["data"]

        # # print(data_list)

        for index, li in enumerate(data_list):

            scr = li["attributes"]["image"]["large"]
            # title = li["attribute"]["slug"]
            
            file = f"{path}pexel_{key_word}_{index+ (page-1)*24}.jpg"

            try:
                time.sleep(1)
                re = requests.get(scr)

                with open(file, "wb") as f:
                    f.write(re.content)

            except Exception as e:
                print(f"{index}:{e}")


from concurrent import futures

def download_start(end_page):
    workers = 100
    with futures.ThreadPoolExecutor(workers) as e:
        e.map(get_url_down, [page_num for page_num in range(end_page)])

download_start(1000)
