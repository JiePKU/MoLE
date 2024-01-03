
import requests
from pyquery import PyQuery as pq
 
count = 1
 
def download_url_images(page):
    """
    :param page:
    :return:
    """
    global count
 
    url='https://gratisography.com/page/{}/'.format(page_number)
 
    headers={
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'
    }
 
    response_data=requests.get(url,headers=headers).text
    #print(response_data)

    doc=pq(response_data)
    #print(doc)
 
    single=doc('.single-photo-thumb a img').items()

    for i in single:
        img_url=i.attr('src')
        img_data=requests.get(img_url,headers=headers)
        with open('graph/{}.jpg'.format(count),'ab',) as f:
            f.write(img_data.content)
            count +=1
 
if __name__=="__main__":
    page_number=int(input("input the number of pages:"))
    for page in (1,page_number):
        download_url_images(page)