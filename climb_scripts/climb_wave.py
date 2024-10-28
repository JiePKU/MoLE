from bs4 import BeautifulSoup
import requests
gHeads = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36"
}
for i in range(1,84):
    url = "https://wallhaven.cc/search?q=woman&page=%s"%(i)
    print(url)
    html = requests.get(url,headers=gHeads)
    html = html.content
    soup = BeautifulSoup(html, 'lxml')
    href_all = soup.find_all( 'a',{"class": "preview"})
    for href in href_all:
        href_url = href['href']
        html4 = requests.get(href_url,headers=gHeads).content
        soup4 = BeautifulSoup(html4, 'lxml')
        img4 = soup4.find( 'img',{"id": "wallpaper"})
        print(img4)
        urlimg = img4['src']
        r = requests.get(urlimg, stream=True)
        image_name = urlimg.split('/')[-1]
        with open('./wave/%s' % image_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=128):
                f.write(chunk)
        print('Saved %s' % image_name)
    print("end.....................")    


