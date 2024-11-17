import pandas as pd
import requests
import os
import zipfile


csv_file = '/Users/zhanghongzun/PycharmProjects/skin/fitzpatrick17k.csv'
df = pd.read_csv(csv_file)


output_dir = 'images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


image_urls = df['url'].dropna()  # get url
#image_urls = df['url'].dropna().iloc[:30]

image_paths = []
for i, url in enumerate(image_urls):
    try:
        #
        response = requests.get(url, headers=headers)
        response.raise_for_status()


        image_name = f'image_{i + 1}.jpg'
        image_path = os.path.join(output_dir, image_name)

        # save image
        with open(image_path, 'wb') as file:
            file.write(response.content)

        #
        print(f"Image {image_name} downloaded and saved successfully.")

        image_paths.append(image_path)

    except requests.exceptions.RequestException as e:
        print(f"Failed to download image {url}: {e}")

# get zip
zip_filename = 'images.zip'
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for image_path in image_paths:
        zipf.write(image_path, os.path.basename(image_path))  #

print(f"All images are compressed into {zip_filename}")