# coding=utf-8
import requests as rq
import json

if __name__ == '__main__':
    url = "http://47.102.118.1:8089/api/challenge/submit"
    data={"uuid": "ac09e60d-4089-44d6-94af-xxxxxxx",
          "teamid": 43,
          "token": "db89ae69-2536-4f96-91b0-d1c998a7850b",
          "answer": {
                   "operations": "",
                   "swap": []
                   }
          }
    data = json.dumps(data)  # 有的时候data需要时json类型的
    headers = {'content-type': "application/json"}
    #headers = {'content-type': application / json}  # 一种请求头，需要携带
    resp = rq.post(url=url,data=data,headers=headers)
    a=json.loads(resp.text)
    b=a.get("data")
    print("作者: ", b.get("owner"))
    print("排名: ", b.get("rank"))
    print("步数: ", a.get("step"))
    print("是否成功: ", a.get("success"))
    print("时间: ", a.get("timeelapsed"))
