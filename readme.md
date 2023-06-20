### 1. 使用代理时，会报错check_hostname requires server_hostname
```shell
# 使用较低版本的urllib3
# 通过pip命令pip list查看当前的urllib3版本；如果其版本大于1.25.7，则将其卸载，再重新安装1.25.7的urllib3；
$ pip uninstall urllib3
$ pip install urllib3==1.25.7
```