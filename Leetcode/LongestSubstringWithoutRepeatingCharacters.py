class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        p1, p2 = 0, 0
        max_len = 0
        while p1 <= len(s) and p2 <= len(s):
            if len(set(s[p1:p2])) < (p2 - p1):
                p1 += 1
            else:
                if (p2 - p1) >  max_len:
                    max_len = p2 - p1
                p2 += 1

        return max_len