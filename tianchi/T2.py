#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/11 9:59                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

def fun(nums):

    if(len(nums) <= 1):
        return len(nums)
    ans = 1
    count = 1
    for i in range(len(nums)-1):
        if (nums[i + 1] > nums[i]):
            count+=1
        else:
            count = 1

        ans = count if count>ans else ans
    return ans


nums = [1,2,3,5,4,10,6,8,7]
# print(fun(nums))


# 快速排序是一种原地排序，只需要一个很小的栈作为辅助空间，空间复杂度为O(log2n)，所以适合在数据集比较大的时候使用。
# 时间复杂度比较复杂，最好的情况是O(n)，最差的情况是O(n2)，所以平时说的O(nlogn)，为其平均时间复杂度。
def quick_sort(arr,l,r):
    if l<r:
        i=l
        j=r
        x=arr[i]
        while i<j:
            while (i < j and arr[j] > x):
                j -= 1
            if i<j:
                arr[i]=arr[j]
                i+=1
            while (i<j and arr[i]<x):
                i +=1
            if i<j:
                arr[j] = arr[i]
                j-=1

        arr[i] = x
        quick_sort(arr,l,i-1)
        quick_sort(arr,i+1,r)

# quick_sort(nums,0,len(nums)-1)
# print(nums)

def select_sort(a):
    i=0 # 有序区的末尾位置
    j=0 #无序区的起始位置
    m=0# 无序区中最小元素位置
    for i in range(len(a)):
        m = i
        #找出"a[i+1] ... a[n]"之间的最小元素，并赋值给min。
        for j in range(i+1,len(a)):
            if a[j]<a[m]:
                m = j

        # 若min != i，则交换 a[i] 和 a[min]。
        # 交换之后，保证了a[0]...a[i] 之间的元素是有序的。
        if (m != i):
            # swap(a[i], a[m])
            pass

