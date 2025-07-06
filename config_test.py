# -i https://pypi.tuna.tsinghua.edu.cn/simple/
# D:\g3\python3.8\python.exe -m pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple/
import sys
print("当前Python路径:", sys.executable)

# # 测试所有包
# try:
#     import pandas as pd
#     print("pandas版本:", pd.__version__)
# except ImportError as e:
#     print("pandas导入失败:", e)
#
# try:
#     import numpy as np
#     print("numpy版本:", np.__version__)
# except ImportError as e:
#     print("numpy导入失败:", e)
#
# try:
#     import matplotlib.pyplot as plt
#     print("matplotlib版本:", plt.matplotlib.__version__)
# except ImportError as e:
#     print("matplotlib导入失败:", e)

# try:
#     import seaborn as sns
#     print("seaborn版本:", sns.__version__)
#     print(" seaborn安装成功！")
# except ImportError as e:
#     print("seaborn导入失败:", e)
#
# try:
#     import jieba
#     print("jieba 版本：", jieba.__version__)
#     print("jieba 安装成功")
# except ImportError as e:
#     print("jieba导入失败：", e)
#

try:
    import tqdm
    print("tqdm 安装成功")
except ImportError as e:
    print("tqdm导入失败：", e)
