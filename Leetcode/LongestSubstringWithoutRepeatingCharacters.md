## Initial Version

Sliding Window + Unordered Map
```C
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int max = 0;
        int count = 0;
        unordered_map<char, int> m;
        int i = 0;
        while(i < s.length()){
            if(i == 0) {
                count = 1;
                max = 1;
                m.insert(pair<char, int>(s.at(i), i));
                ++i;
                continue;
            }
            if(m.find(s.at(i)) != m.end()) {
                count = 1;
                auto iter = m.find(s.at(i));
                i = iter->second + 1;
                m.clear();
                m.insert(pair<char, int>(s.at(i), i));
            }
            else {
                ++count;
                m.insert(pair<char, int>(s.at(i), i));
            }
            if(count > max) max = count;
            ++i;
            
        }
        return max;
    }
};
```

## Revised Version

Hash Map
```C
/**
 * Solution (DP, O(n)):
 * 
 * Assume L[i] = s[m...i], denotes the longest substring without repeating
 * characters that ends up at s[i], and we keep a hashmap for every
 * characters between m ... i, while storing <character, index> in the
 * hashmap.
 * We know that each character will appear only once.
 * Then to find s[i+1]:
 * 1) if s[i+1] does not appear in hashmap
 *    we can just add s[i+1] to hash map. and L[i+1] = s[m...i+1]
 * 2) if s[i+1] exists in hashmap, and the hashmap value (the index) is k
 *    let m = max(m, k), then L[i+1] = s[m...i+1], we also need to update
 *    entry in hashmap to mark the latest occurency of s[i+1].
 * 
 * Since we scan the string for only once, and the 'm' will also move from
 * beginning to end for at most once. Overall complexity is O(n).
 *
 * If characters are all in ASCII, we could use array to mimic hashmap.
 */

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        vector<int> dict(256, -1);
        int maxlen = 0;
        int m = 0;
        for(int i = 0; i < s.length(); ++i)
        {
            m = max(dict[s[i]] + 1, m);
            dict[s[i]] = i;
            maxlen = max(maxlen, i - m + 1);
        }
        return maxlen;
    }
};
```

## Notes

Differences between **std::map** and **std::unordered_map**

- map
    - 优点：有序性，这是map结构最大的优点，其元素的有序性在很多应用中都会简化很多的操作，内部实现一个红黑书使得map的很多操作在lgn的时间复杂度下就可以实现，因此效率非常的高
    - 缺点：空间占用率高，因为map内部实现了红黑树，虽然提高了运行效率，但是因为每一个节点都需要额外保存父节点、孩子节点和红/黑性质，使得每一个节点都占用大量的空间
    - 适用处：对于那些有顺序要求的问题，用map会更高效一些

- unordered_map
    - 优点：因为内部实现了哈希表，因此其查找速度非常的快
    - 缺点：哈希表的建立比较耗费时间
    - 适用处：对于查找问题，unordered_map会更加高效一些，因此遇到查找问题，常会考虑一下用