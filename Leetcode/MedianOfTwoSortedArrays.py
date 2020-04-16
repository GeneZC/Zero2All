class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        len1 = len(nums1)
        len2 = len(nums2)
        pivot = (len1 + len2) // 2
        odd = (len1 + len2) % 2
        if len1 == 0:
            if odd:
                return nums2[pivot]
            else:
                return (nums2[pivot] + nums2[pivot-1]) / 2.0
        if len2 == 0:
            if odd:
                return nums1[pivot]
            else:
                return (nums1[pivot] + nums1[pivot-1]) / 2.0
        prev = []
        p1, p2 = 0, 0
        while True:
            if len(prev) - 1 == pivot:
                if odd:
                    return prev[-1]
                else:
                    return (prev[-1] + prev[-2]) / 2.0
            try:
                num1 = nums1[p1]
            except:
                num1 = float('inf')
            try:
                num2 = nums2[p2]
            except:
                num2 = float('inf')
            if num1 >= num2:
                prev.append(num2)
                p2 += 1
            else:
                prev.append(num1)
                p1 += 1

