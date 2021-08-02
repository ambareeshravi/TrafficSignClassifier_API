
'''
Author: Ambareesh Ravi
Date: Jul 31, 2021
Title: api_test.py
Description:
    Tests the working of the API
'''

import requests

if __name__ == '__main__':
    URL = "http://127.0.0.1:12345/predict"
    IMAGE_PATH = "./TrafficSign_Images/test/00001.png"

    image = open(IMAGE_PATH, "rb").read()
    payload = {"image": image}

    r = requests.post(URL, files=payload).json()

    if r["success"]:
        print(r)
    else:
        print("Request failed")