from bs4 import BeautifulSoup
import requests
gHeads = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36"
}
label = ['adult', 'child', 'person', 'people', 'girl', 'boy', 'woman', 'man', 'kid' 'elderly', 'teenager', 'baby']
def get_url_down(page):
    i = page
    # key = "adult"
    for key in ['hand']:
        url = f"https://www.hippopx.com/zh/query?q={key}&page=%s"%(i)
        print(url)
        html = requests.get(url,headers=gHeads)
        html = html.content
        soup = BeautifulSoup(html, 'lxml')
        img_all = soup.find_all('img',{"itemprop": "contentUrl"})
        for img in img_all:
            urlimg = img['src']
            r = requests.get(urlimg, stream=True)
            image_name = urlimg.split('/')[-1]
            with open('./hot_hand/%s_%s' %(page, image_name), 'wb') as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)
            # print('Saved %s' % image_name)
    print(f"page {page} end.....................")

from concurrent import futures
def download_start(end_page):
    workers = 100
    with futures.ThreadPoolExecutor(workers) as e:
        e.map(get_url_down, [page_num for page_num in range(2, end_page)])


download_start(1000)

