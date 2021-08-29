# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

x = [1, 2, 5]
y = 10
a = [0, 0, 0]
b = 0

nList = []
zList = []
a1List = []
a2List = []
a3List = []
bList = []

def gd():
    global x, y, a, b
    z = a[0] * x[0] + a[1] * x[1] + a[2] * x[2] + b
    err = (z - y) * 0.01
    a[0] -= err * x[0]
    a[1] -= err * x[1]
    a[2] -= err * x[2]
    b -= err

if __name__=="__main__":

    for i in range(10):
        # nの値
        nList.append(i)

        # zの値
        z = a[0] * x[0] + a[1] * x[1] + a[2] * x[2] + b
        zList.append(z)

        # aの値
        a1List.append(a[0])
        a2List.append(a[1])
        a3List.append(a[2])
        bList.append(b)
        
        # 勾配降下法
        gd()
        

    # パラメータの推移
    plt.plot(nList, a1List, label='a1')
    plt.plot(nList, a2List, label='a2')
    plt.plot(nList, a3List, label='a3')
    plt.plot(nList, bList, label='b')
    plt.legend()
    plt.show()

    # 出力の推移
    plt.plot(nList, zList, label='z(output)')
    plt.hlines(y=10, xmin=0, xmax=nList[-1])
    plt.legend()
    plt.show()

    print('a1:', a[0])
    print('a2:', a[1])
    print('a3:', a[2])
    print('b:', b)
    print('z(output):', z)
