#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/4 10:23                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 对排序
# 1. 建堆(最大堆，最小堆)，最大堆常用升序排序，最小堆常用于降序排序
# 2. 调整+数据交换。


# 最大堆的向下调整法
def adjust_heap_max1(heap,start,end):
    current = start
    left = 2*current+1
    tmp = heap[current]

    while left <=end:
        if (left < end) and (heap[left]<heap[left+1]):
            left = left+1
        if tmp>=heap[left]:
            break #
        else:
            heap[current] = heap[left]# 将根节点的数据根最大值的孩子节点的数据交换
            heap[left] = tmp

        current = left
        left = 2 * current + 1

def adjust_heap_max2(heap,start,end):
    left = 2*start+1
    right = left+1
    larger = start

    if left <= end and heap[larger] < heap[left]:
        larger = left
    if right <= end and heap[larger] < heap[right]:
        larger = right
    if larger != start:
        heap[larger],heap[start] = heap[start],heap[larger]
        adjust_heap_max2(heap,larger,end)


#

def heap_sort(heap):
    # 1.构建最大堆
    for i in range(len(heap)//2-1,-1,-1):
        # adjust_heap_max1(heap,i,len(heap)-1)
        adjust_heap_max2(heap,i,len(heap)-1)

    print('构建完成后',heap)

    # 2. 从最后一个元素开始对序列进行调整，不断的缩小调整的范围直到第一个元素
    for i in range(len(heap)-1,0,-1):
        # 交换a[0]和a[i],交换后，a[i]是a[0,....i]中最大的
        heap[0],heap[i] = heap[i],heap[0]
        # 调整a[0....i-1],使得a[0....i-1]仍然是一个最大堆
        # 即保证a[i-1]是a[0...i-1]中的最大值
        # adjust_heap_max1(heap,0,i-1)
        adjust_heap_max2(heap,0,i-1)

if __name__ == '__main__':
    heap = [20, 30, 90, 40, 70, 110, 60, 10, 100, 50, 80]
    print('before sort',heap)
    heap_sort(heap)
    print('after sort',heap)