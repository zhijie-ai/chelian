# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/11/8 15:29
# @User   : RANCHODZU
# @File   : utils.py
# @e-mail : zushoujie@ghgame.cn
def find_selected_item_v1(dp, w):
    # 逆向查找确定所选物品
    max_value = dp[-1][-1]
    selected = []
    j = len(dp[0])
    for i in range(len(dp)-1, 0, -1):
        if max_value == dp[i - 1][j]:
            continue
        else:
            selected.append(i - 1)  # 物品索引从0开始，输出时加1
            j -= w[i - 1]
    selected.reverse()
    return dp[-1][-1], selected


# 这种方法效率更高
# 最大价值情形的回溯
def find_selected_item_v2(dp, w):

    # 逆向查找确定所选物品
    max_value = dp[-1][-1]
    j = dp[-1].index(max_value)
    selected = []
    for i in range(len(dp)-1, 0, -1):
        if max_value == dp[i - 1][j]:
            continue
        else:
            selected.append(i - 1)  # 物品索引从0开始，输出时加1
            j -= w[i - 1]
    selected.reverse()
    print(f"最大价值: {dp[-1][-1]}")
    print(f"所选物品索引: {selected}")
    return dp[-1][-1], selected


def get_dp_array(n, W, w, v):
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    # 填表
    for i in range(1, n + 1):
        for j in range(W + 1):
            if j >= w[i - 1]:
                dp[i][j] = max(dp[i - 1][j], v[i - 1] + dp[i - 1][j - w[i - 1]])
            else:
                dp[i][j] = dp[i - 1][j]

    return dp


if __name__ == '__main__':
    # 示例
    # n_ = 5
    # W_ = 10
    # w_ = [2, 3, 4, 5, 6]
    # v_ = [3, 4, 5, 6, 7]

    w_ = [1, 3, 4]
    v_ = [15, 20, 30]
    W_ = 5
    n_ = len(w_)

    dp = get_dp_array(n_, W_, w_, v_)
    print(dp)
    max_v, selected_items = find_selected_item_v2(dp, w_)
    print(f"最大价值: {max_v}")
    print(f"所选物品索引: {selected_items}")