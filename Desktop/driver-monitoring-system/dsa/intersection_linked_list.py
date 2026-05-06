class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def getIntersectionNode(self, headA, headB):
        A = headA
        B = headB

        while A != B:
            A = headB if A is None else A.next
            B = headA if B is None else B.next

        return A


# -------- TEST --------
common = ListNode(8)
common.next = ListNode(10)

headA = ListNode(3)
headA.next = ListNode(7)
headA.next.next = common

headB = ListNode(99)
headB.next = common

sol = Solution()
res = sol.getIntersectionNode(headA, headB)

if res:
    print("Intersection at:", res.val)
else:
    print("No intersection")