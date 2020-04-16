# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode(0)
        ptr = head
        l1_val = l1.val
        l2_val = l2.val
        pivot = 0
        flag1, flag2 = 0, 0
        while True:
            curr_sum = l1_val + l2_val + pivot
            if curr_sum > 9:
                ptr.val = curr_sum % 10
                pivot = 1
            else:
                ptr.val = curr_sum
                pivot = 0
            try:
                l1 = l1.next
                l1_val = l1.val
            except:
                l1_val = 0
                flag1 = 1
            try:
                l2 = l2.next
                l2_val = l2.val
            except:
                l2_val = 0
                flag2 = 1
            if flag1 and flag2:
                if pivot:
                    ptr.next = ListNode(pivot)
                    ptr = ptr.next
                break
            ptr.next = ListNode(pivot)
            ptr = ptr.next
            
        return head