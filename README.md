# hetero-secure-boost-origin
对着FATE V1.0版本的hetero secure boost模块写的，算法模块来自FATE，底层计算和联邦模块自写，方便本地调试和运行算法。

## 依赖
```
gmpy2==2.0.8
llvmlite==0.36.0
numba==0.53.1
numpy==1.21.1
protobuf==3.17.3
pycryptodomex==3.10.1
six==1.16.0
```

## 启动示例

### guest 方
```
python .\guest.py data/breast_hetero/breast_hetero_guest.csv 1 0.8 CLASSIFICATION 10086
```

### host 方
```
python .\host.py data/breast_hetero/breast_hetero_host.csv 0.8 10086 0
```

<!--requirements.txt 里的gmpy2可以在[这里](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gmpy)下载对应版本的wheel，然后在pip install 这个wheel

requirements.txt 里的protobuf，可以在[这里](https://github.com/protocolbuffers/protobuf/releases)下载对应版本的protobuf和protoc，然后参考[这篇博客](https://blog.csdn.net/chenkjiang/article/details/22159407/)把这个装上-->

## TODO

当前版本是针对 FATE v1.0 写的，并没有跟上后期的大部分优化，这些暂时作为后期的计划。

* 分裂点密文压缩算法
* 梯度打包策略
* 直方图计算优化
* 尽量少地使用python原生大数计算操作
* GOSS采样