"""
https://www.logosc.cn/so/
target: https://www.logosc.cn/api/so/get?page=0&pageSize=20&keywords=&category=local&isNeedTranslate=undefined
"""
import requests
import os.path

headers = {
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
}

# label = ['adult', 'child', 'person', 'people', 'girl', 'boy', 'woman', 'man', 'kid' 'elderly', 'teenager', 'baby', 'mother', 'father']
label = ['hand']

def getPicture(page):
    for key_word in label:
        url = f"https://www.logosc.cn/api/so/get?category=pixabay&isNeedTranslate=false&keywords={key_word}&page={page}&pageSize=20"
        print(url)
        response = requests.get(url=url, headers=headers)
        content = response.json()
        if "data" in content and "source" in content:
            if content["source"] == "pixabay":
                i = 0
                while True:
                    try:
                        if content["data"][i]["large_img_path"]["url"]:
                            picture_url = content["data"][i]["large_img_path"]["url"]
                            image = requests.get(picture_url).content
                            with open(f'log_hand/{key_word}_{page}_'+str(i) + '.jpg', 'wb') as f:  
                                f.write(image)
                            i += 1

                    except:
                        print("no data")
                        break     
        else:
            print("do not obtain data!")


from concurrent import futures
def download_start(end_page):
    workers = 100
    with futures.ThreadPoolExecutor(workers) as e:
        e.map(getPicture, [page_num for page_num in range(end_page)])

download_start(1000)

