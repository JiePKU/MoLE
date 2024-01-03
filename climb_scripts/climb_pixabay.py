
import os
from PIL import Image
import requests

with open('pixababy.txt','r') as f:
    contents = f.read()

    contents = contents.split('\n')
    # print(contents[-1])

    used = [i for i in contents if 'pixabay' in i and 'jpg' in i]
    print(len(used))

    print(used[-1])
    saved_file_names = os.listdir('./pixabay/')

    for i, picture_url in enumerate(used):
        basename = os.path.basename(picture_url).split('.')[0]
        if (f'{basename}_'+str(i) + '.jpg' in saved_file_names):
            continue
        image = requests.get(picture_url).content
        with open(f'pixabay/{basename}_' + str(i) + '.jpg', 'wb') as f: 
            f.write(image)