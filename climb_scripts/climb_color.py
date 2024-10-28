from bs4 import BeautifulSoup
import requests
import re
from io import BytesIO
from PIL import Image
gHeads = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36"
}
# label = ['adult', 'child', 'person', 'people', 'girl', 'boy', 'woman', 'man', 'kid' 'elderly', 'teenager', 'baby', 'mother', 'father']

label = ['hand']
def get_url_down(key):
    pages = 48
    # key = "adult"
    for i in range(1,pages):
        url = f"https://www.colorhub.me/search?tag={key}&page=%s"%(i)
        print(url)
        html = requests.get(url,headers=gHeads)
        html = html.content
        soup = BeautifulSoup(html, 'lxml')
        href_all = soup.find_all('div',{"class":"card"})
        print(len(href_all))
        for href in href_all:
            try:
                href_url = href.a['href']
                html4 = requests.get(href_url,headers=gHeads).content
                soup4 = BeautifulSoup(html4, 'lxml')
                img4 = soup4.find('a',{"data-magnify":"gallery"})
                urlimg ="http:"+img4['href']
                r = requests.get(urlimg, stream=True)
                image_name = urlimg.split('/')[-1].replace('webp', 'jpg')
                byte_stream = BytesIO(r.content)
                im = Image.open(byte_stream)
                # im.show()
                if im.mode == "RGBA":
                    im.load()  # required for png.split()
                    background = Image.new("RGB", im.size, (255, 255, 255))
                    background.paste(im, mask=im.split()[3])  
                im.save('./color_hand/{}'.format(image_name), 'JPEG')
            # print('Saved %s' % image_name)
            except:
                continue
        print(f"page {key} {i} end.....................") 


from concurrent import futures
def download_start(end_page):
    workers = 100
    with futures.ThreadPoolExecutor(workers) as e:
        e.map(get_url_down, [page_num for page_num in label])


download_start(1000)

# get_url_down()