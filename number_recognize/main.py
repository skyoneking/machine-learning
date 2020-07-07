
import numpy as np

from input import init_data, X_test, Y_test
from BP import BP
import sys
sys.setrecursionlimit(1000000)

K = 10

X, Y, X_valid, Y_valid, X_test, Y_test = init_data(K=K)
# X, Y, X_valid, Y_valid, X_test, Y_test = init_data(X_test, Y_test, K=K)

bp = BP(X, Y, X_valid=X_valid, Y_valid=Y_valid, X_test=X_test, Y_test=Y_test, K=10, iter_num=1000, hidden_num=2,
        hidden_cell_num=100, alpha=0.3, labd=0.006, threshold=0, is_save_module=False, is_use_module=False)

# bp.check_gd() # 梯度检验
bp.fit()
# bp.predict()
# bp.predict(X_test, Y_test, text='acc_test')
# bp.learn_curve()
