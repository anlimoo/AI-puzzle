# coding=utf-8
import requests as rq
import json

if __name__ == '__main__':
    url = "http://47.102.118.1:8089/api/challenge/create"
    data={"teamid": 43,
          "data": {
                    "letter": "a",
                    "exclude": 5,
                    "challenge": [
                                  [1, 2, 3],
                                  [0, 4, 6],
                                  [7, 8, 9]
                                  ],
                    "step": 20,
                    "swap": [1,2]
                    },
          "token": "db89ae69-2536-4f96-91b0-d1c998a7850b"}
    data = json.dumps(data)  # 有的时候data需要时json类型的
    headers = {'content-type': "application/json"}
    #headers = {'content-type': application / json}  # 一种请求头，需要携带
    resp = rq.post(url=url,data=data,headers=headers)
    a=json.loads(resp.text)
    b=a.get("data")
    print("信息: ", b.get("message"))
    print("是否成功: ", b.get("success"))
    print("题目标识: ", a.get("uuid"))
