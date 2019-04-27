## Initial Version

Simple Combination
```C
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int x, y;
        vector<int> indices;
        bool flag = false;
        size_t size = nums.size();
        for (int i = 0; i < size; ++i) {
            for (int j = i + 1; j < size; ++j) {
                if (nums[i] + nums[j] ==  target) {
                    indices.emplace_back(i);
                    indices.emplace_back(j);
                    flag = true;
                }
                if (flag) break;
            }
            if (flag) break;
        }
        return indices;
    }
};
```

## Revised Version

Hash Map
```C
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> indices;
        map<int, int> hash_map;
        size_t size = nums.size();
        for(int i = 0; i < size; ++i) {
            if (hash_map.find(target - nums[i]) != hash_map.end()) {
                indices.emplace_back(hash_map[target - nums[i]]);
                indices.emplace_back(i);
                break;
            }
            hash_map.insert(pair<int, int>(nums[i], i));
        }
        return indices;
    }
};
```

## Notes

A example about **std::map** in C++

```C
// init
map<int, int> hash_map;
// insert
hash_map.insert(pair<int, int>(1, 2));
// find
auto iter = hash_map.find(1);
// existence
if (iter != hash_map.end()) {
    // exists
} 
else {
    // doesn't exist
}
// erase, two ways
hash_map.erase(hash_map.begin(), hash_map.end());
hash_map.erase(iter);
hash_map.erase(1);
// clear
hash_map,clear();
```