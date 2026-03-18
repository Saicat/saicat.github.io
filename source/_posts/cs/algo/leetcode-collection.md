---
title: LeetCode面试题解整理
tags:
  - leetcode
  - algorithm
categories:
  - CS
  - algorithm
  - programming
  - python
abbrlink: dd5035ac
date: 2022-02-21 12:18:17
---

# 高频基础题

---

## LeetCode 88 - 合并两个有序数组

**原题描述：**
- 给定两个按非递减顺序排列的整数数组 `nums1` 和 `nums2`，将 `nums2` 合并到 `nums1` 中，使合并后的数组同样按非递减顺序排列。
- `nums1` 的初始长度为 `m + n`，前 `m` 个元素是有效数据，后 `n` 个为 0。
- 结果必须存储在 `nums1` 中，时间复杂度 O(m + n)。

**解法：双指针（从后向前）**

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p, p1, p2 = m + n - 1, m - 1, n - 1
        while p >= 0:
            if p2 < 0:  # p2的数字都已经排完了，而p1都是已经在num1排好的，不用动
                break
            elif p1 < 0:
                nums1[p] = nums2[p2]
                p2 -= 1
                # p1已经排好，只要把p2剩余的放上去就行，更直接的做法可以：
                # nums1[:p + 1] = nums2[:p2 + 1]
                # break
            elif nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
            p -= 1
```

- **时间复杂度：** O(m + n)，每个元素最多被访问一次
- **空间复杂度：** O(1)，原地修改

---

## LeetCode 5 - 最长回文子串

**原题描述：**
- 给定字符串 s，找到 s 中最长的回文子串。回文串正读和反读都相同。
- 示例："babad" → "bab" 或 "aba"；"cbbd" → "bb"

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n < 2:
            return s
        
        ans_start = 0
        ans_len = 1
        dp = [[False] * n for _ in range(n)]
        
        for i in range(n):
            dp[i][i] = True
        
        for length in range(2, n + 1):
            for i in range(n):
                # length = j - i + 1
                j = i + length - 1
                if j >= n:
                    break
                
                if s[i] != s[j]:
                    dp[i][j] = False
                else:
                    if length <= 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                
                if dp[i][j] and length > ans_len:
                    ans_start = i
                    ans_len = length
        
        return s[ans_start: ans_start + ans_len]
```

- **时间复杂度：** O(n²)
- **空间复杂度：** O(n²)

---

## LeetCode 542 - 01 矩阵

**原题描述：**
- 给定由 0 和 1 组成的矩阵，求每个单元格到最近的 0 的距离。相邻单元格（上下左右）距离为 1。

**解法：多源 BFS**

**详细思路**

1. **直觉**：每个 1 要找"最近的 0"，等价于从所有 0 一起出发，看谁先"扩散"到这个 1。扩散的层数就是距离。

2. **多源 BFS 的物理类比**：想象把 0 的位置都滴上一滴水，水会同时向四周扩散。某个 1 第一次被水到达时，它离最近 0 的距离就是扩散的"层数"。

3. **做法**：
   - 把所有 0 的坐标放进队列，距离设为 0
   - 1 的位置先设为无穷大（表示还没被扩散到）
   - 从队列取格子，看它上下左右四个邻居；如果"当前距离+1"能更新邻居（让邻居距离变小），就更新邻居并把它入队
   - 这样每个格子第一次被更新时，得到的是一定是最短距离

4. **为什么不会错？** BFS 保证按距离递增访问，所以第一次到达某格时的距离就是最短距离。

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        q = deque()
        ans = [[float("inf")] * n for _ in range(m)]
        visited = set()
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    visited.add((i, j))
                    ans[i][j] = 0
                    q.append((i, j))
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        while q:
            i, j = q.popleft()
            for neighbor in neighbors:
                ni, nj = i + neighbor[0], j + neighbor[1]
                if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in visited:
                    ans[ni][nj] = ans[i][j] + 1
                    visited.add((ni, nj))
                    q.append((ni, nj))
        return ans
```

- **时间复杂度：** O(m×n)，每个格子最多入队一次
- **空间复杂度：** O(m×n)，队列空间

也可以在原mat上修改，而不需要使用新的空间：

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        q = deque()
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    q.append((i, j))
                else:
                    mat[i][j] = float("inf")
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        while q:
            i, j = q.popleft()
            for neighbor in neighbors:
                ni, nj = i + neighbor[0], j + neighbor[1]
                if 0 <= ni < m and 0 <= nj < n and mat[ni][nj] > mat[i][j] + 1:
                    mat[ni][nj] = mat[i][j] + 1
                    q.append((ni, nj))
        return mat
```

- **时间复杂度：** O(m×n)，每个格子最多入队一次
- **空间复杂度：** O(1)，没有使用额外空间

这里还有另外一个变化点，就是不需要使用visited set了，因为这里BFS本身就隐藏了一个情况：未访问的位置的值为inf，那么当它第一次被访问到的时候，此时的值肯定比后续访问它的时候要更小（近的线扩散到，远的还没扩散到，那么后访问它的肯定是离得远的，因此值肯定更大），因此可以直接对比原值和新值的大小进行判断是否要更新即可。

---

## LeetCode 215 - 数组中的第 K 个最大元素

**原题描述：**
- 在未排序数组中找到第 k 大的元素（排序后的第 k 大）。
- 示例：[3,2,1,5,6,4], k=2 → 5

**解法：最小堆** 维护大小为 k 的最小堆，堆顶是当前最小的"候选第 k 大"。遍历数组，若当前数比堆顶大就替换堆顶；最后堆顶就是第 k 大。O(n log k)。

**最小堆解法：**

```python
import heapq

def findKthLargest(nums: List[int], k: int) -> int:
    """维护大小为 k 的最小堆，堆顶即为第 k 大"""
    heap = nums[:k]
    heapq.heapify(heap)  # 最小堆
    for x in nums[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)  # 弹出堆顶，加入 x
    return heap[0]
```

时间复杂度O(nlogk)。

或者直接把整个list变成堆，然后pop到第k个：

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums = [-num for num in nums]
        heapq.heapify(nums)
        for i in range(k):
            ans = heapq.heappop(nums)
        return -ans
```

- 时间复杂度：O(nlogn)，建堆的时间代价是 O(n)，删除的总代价是 O(klogn)，因为 k < n，故渐进时间复杂为 O(n+klogn)=O(nlogn)。
- 空间复杂度：O(logn)，即递归使用栈空间的空间代价。

---

## LeetCode 72 - 编辑距离

**原题描述：**
- 给定两个单词 word1 和 word2，计算将 word1 转换成 word2 所需的最少操作数。
- 允许操作：插入一个字符、删除一个字符、替换一个字符。

**解法：动态规划**

**详细思路（新手向）：**

1. **状态定义**：`dp[i][j]` = 把 word1 的前 i 个字符变成 word2 的前 j 个字符，最少需要几步。这样最终答案就是 `dp[m][n]`。

2. **最后一个字符相同**：若 word1[i-1] == word2[j-1]，最后一位不用动，`dp[i][j] = dp[i-1][j-1]`。

3. **最后一个字符不同**：可以选三种操作之一，取步数最少的：
   - **删除** word1[i-1]：剩下 word1[:i-1] 要变成 word2[:j]，即 `dp[i-1][j] + 1`
   - **插入** word2[j-1]：等价于 word1[:i] 先变成 word2[:j-1]，再插入，即 `dp[i][j-1] + 1`
   - **替换**：把 word1[i-1] 换成 word2[j-1]，即 `dp[i-1][j-1] + 1`

4. **边界**：空串变成长度 j 的串要 j 次插入；长度 i 的串变成空串要 i 次删除。

5. **填表顺序**：从左到右、从上到下，因为 `dp[i][j]` 只依赖左、上、左上。

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[-1][-1]
```

- **时间复杂度：** O(m×n)
- **空间复杂度：** O(m×n)，可滚动数组优化为 O(n)

---

## LeetCode 54 - 螺旋矩阵

**原题描述：**
- 给定 m×n 矩阵，按顺时针螺旋顺序返回所有元素。
- 示例：[[1,2,3],[4,5,6],[7,8,9]] → [1,2,3,6,9,8,7,4,5]

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m, n = len(matrix), len(matrix[0])
        visited = set()
        ans = [0] * (m * n)
        cur_i, cur_j = 0, 0
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dir_idx = 0

        for i in range(len(ans)):
            ans[i] = matrix[cur_i][cur_j]
            visited.add((cur_i, cur_j))
            next_i, next_j = cur_i + dirs[dir_idx][0], cur_j + dirs[dir_idx][1]
            if not (m > next_i >= 0 and n > next_j >= 0 and (next_i, next_j) not in visited):
                dir_idx += 1
                dir_idx %= 4
            cur_i, cur_j = cur_i + dirs[dir_idx][0], cur_j + dirs[dir_idx][1]
        return ans
```

---

## LeetCode 59. 螺旋矩阵 II

**原题描述：**
- 给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。
- 示例：输入：n = 3，输出：[[1,2,3],[8,9,4],[7,6,5]]

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        ans = [[0] * n for _ in range(n)]
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dir_idx = 0
        cur_i, cur_j = 0, 0
        visited = set()
        for i in range(n * n):
            ans[cur_i][cur_j] = i + 1
            visited.add((cur_i, cur_j))
            next_i, next_j = cur_i + dirs[dir_idx][0], cur_j + dirs[dir_idx][1]
            if not (n > next_i >= 0 and n > next_j >= 0 and (next_i, next_j) not in visited):
                dir_idx += 1
                dir_idx %= 4
            cur_i, cur_j = cur_i + dirs[dir_idx][0], cur_j + dirs[dir_idx][1]
        return ans
```

---

## LeetCode 226 - 翻转二叉树

**原题描述：**
- 翻转二叉树：将每个节点的左右子树交换位置。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return root
        new_left = self.invertTree(root.right)
        new_right = self.invertTree(root.left)
        root.left, root.right = new_left, new_right
        return root
```

**迭代版本（BFS）：**

```python
from collections import deque

def invertTree(root: 'TreeNode') -> 'TreeNode':
    if not root:
        return None
    q = deque([root])
    while q:
        node = q.popleft()
        node.left, node.right = node.right, node.left
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return root
```

- **时间复杂度：** O(n)
- **空间复杂度：** O(h) 递归栈 / O(n) 队列

---

## LeetCode 700 - 二叉搜索树中的搜索

**原题描述：**
- 在 BST 中找到值等于给定值的节点，返回以该节点为根的子树；不存在则返回 null。

**解法：递归**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return None
        
        if root.val == val:
            return root
        
        if val > root.val:
            return self.searchBST(root.right, val)
        
        return self.searchBST(root.left, val)
```

- **时间复杂度：** O(h)，h 为树高
- **空间复杂度：** 迭代 O(1)，递归 O(h)

**解法：利用 BST 性质**

```python
def searchBST(root: 'TreeNode', val: int) -> 'TreeNode':
    """迭代写法：根据 val 与当前节点值的大小关系，选择左或右子树"""
    while root:
        if root.val == val:
            return root
        if root.val > val:
            root = root.left
        else:
            root = root.right
    return None
```

---

## LeetCode 11 - 盛最多水的容器

**原题描述：**
- 给定 n 条垂线高度数组，选两条线与 x 轴构成容器，求最大盛水量。水量 = 宽 × min(左高, 右高)

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        n = len(height)
        left, right = 0, n - 1
        ans = min(height[left], height[right]) * (right - left)
        while left < right:
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
            ans = max(ans, min(height[left], height[right]) * (right - left))
        return ans
```

- **时间复杂度：** O(n)
- **空间复杂度：** O(1)

---

## LeetCode 334 - 递增的三元子序列

**原题描述：**
- 判断数组中是否存在下标 i<j<k 使得 nums[i] < nums[j] < nums[k]。要求 O(n) 时间、O(1) 空间。

思路上更加好理解的方法：从右往左记录最大的数，从左往右记录最小的数，然后再遍历一次，在每个位置对比左边最小和右边最大的数是否符合要求。

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        n = len(nums)

        rmax = [-float("inf")] * n
        for i in range(n - 2, -1, -1):
            rmax[i] = max(rmax[i + 1], nums[i + 1])
        
        lmin = [float("inf")] * n
        for i in range(1, n):
            lmin[i] = min(lmin[i - 1], nums[i - 1])
        
        for i in range(1, n - 1):
            if rmax[i] > nums[i] > lmin[i]:
                return True
        return False
```

或者谈心算法的思路，稍微难理解一点，效率更高：

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        first, second = float("inf"), float("inf")
        for num in nums:
            if num > second:
                return True
            elif num > first:
                second = num
            else:
                first = num
        return False
```

---

## LeetCode 198 - 打家劫舍

**原题描述：**
- 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
- 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

动态规划：

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        
        first, second = 0, nums[0]
        for i in range(1, n):
            tmp = max(first + nums[i], second)
            first, second = second, tmp
        return second
```

---

## LeetCode 213 - 打家劫舍 II

**原题描述：**
- 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
- 给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。

动态规划：

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        def _rob(nums):
            if len(nums) < 1:
                return 0
            first, second = 0, nums[0]
            for i in range(1, len(nums)):
                tmp = max(nums[i] + first, second)
                first, second = second, tmp
            return second
        if len(nums) < 2:
            return max(nums)
        return max(_rob(nums[:-1]), _rob(nums[1:]))
```

---

## LeetCode 337 - 打家劫舍 III

**原题描述：**
- 房屋形成二叉树，相邻节点（直接相连）不能同时打劫。求最大金额。

**解法：树形 DP**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def search(root):
            if not root:
                return 0, 0
            
            left_rob, left_notrob = search(root.left)
            right_rob, right_notrob = search(root.right)
            root_rob = root.val + left_notrob + right_notrob
            root_notrob = max(left_rob, left_notrob) + max(right_rob, right_notrob)
            return root_rob, root_notrob
        return max(search(root))
```

---

## LeetCode 3 - 无重复字符的最长子串

**原题描述：**
- 找不含重复字符的最长子串长度。
- 示例："abcabcbb" → 3（"abc"）

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        d = {}
        left = 0
        ans = 1
        for i, c in enumerate(s):
            if c not in d:
                d[c] = i
            else:
                if d[c] < left:
                    d[c] = i
                else:
                    left = d[c] + 1
                    d[c] = i
            ans = max(ans, i - left + 1)
        return ans
```

---

## LeetCode 121 - 买卖股票的最佳时机

**原题 121：** 最多一次买卖，求最大利润。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2:
            return 0
        minn = prices[0]
        ans = 0
        for i in range(1, len(prices)):
            ans = max(ans, prices[i] - minn)
            minn = min(minn, prices[i])
        return ans
```

---

## LeetCode 122 - 买卖股票的最佳时机 II

给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。然而，你可以在 同一天 多次买卖该股票，但要确保你持有的股票不超过一股。

返回 你能获得的 最大 利润 。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        for i in range(1, len(prices)):
            ans += max(0, prices[i] - prices[i - 1])
        return ans
```

---

## LeetCode 62 - 不同路径

**原题描述：**
- 机器人从左上角 (0,0) 到右下角 (m-1,n-1)，只能向右或向下，求路径数。

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]
```

---

## LeetCode 63 - 不同路径 II

增加了障碍物，有障碍物的各自无法通行。

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            if obstacleGrid[i][0] == 0:
                dp[i][0] = 1
            else:
                break
        for j in range(n):
            if obstacleGrid[0][j] == 0:
                dp[0][j] = 1
            else:
                break
        # print(dp)
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # print(dp)
        return dp[-1][-1]
```

---

## LeetCode 64 - 最小路径和

给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[-1][-1]
```

---

## LeetCode 103 - 二叉树的锯齿形层序遍历

**原题描述：**
- 层序遍历，第一层从左到右，第二层从右到左，交替进行。返回二维列表。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        ans = []
        if not root:
            return ans
        direction = 1
        q = deque()
        q.append(root)
        while q:
            n = len(q)
            layer = []
            for _ in range(n):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
                layer.append(node.val)
            layer = layer[::direction]
            ans.append(layer)
            direction *= -1
        return ans

```

---

## LeetCode 206 - 反转链表

**原题描述：**
- 反转单链表。示例：[1,2,3,4,5] → [5,4,3,2,1]

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre, cur = None, head
        while cur:
            tmp = cur.next
            cur.next = pre
            pre, cur = cur, tmp
        return pre
```

---

## LeetCode 19 - 删除链表的倒数第 N 个结点

**原题描述：**
- 删除倒数第 n 个节点，返回头节点。保证 n 有效。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(next=head)
        slow, fast = dummy, head
        for _ in range(n):
            fast = fast.next
        while fast:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next
```

---

## LeetCode 300 - 最长递增子序列

**原题描述：**
- 求最长严格递增子序列长度。子序列不要求连续。
- 示例：[10,9,2,5,3,7,101,18] → 4（如 [2,3,7,101]）

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n

        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```

---

## LeetCode 1143 - 最长公共子序列

**原题描述：**
- 求两个字符串的最长公共子序列（LCS）长度。子序列不要求连续。

**解法：动态规划**

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                left_down = dp[i - 1][j - 1]
                dp[i][j] = max(left_down + int(text1[i - 1] == text2[j - 1]), dp[i][j - 1], dp[i - 1][j])
        return dp[-1][-1]
```

---

## LeetCode 221 - 最大正方形

**原题描述：**
- 在 '0'/'1' 矩阵中找只含 '1' 的最大正方形面积。

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            for j in range(n):
                matrix[i][j] = int(matrix[i][j])
        dp = [[0] * n for _ in range(m)]
        ans = 0
        for i in range(m):
            dp[i][0] = matrix[i][0]
            ans = max(ans, matrix[i][0])
        for j in range(n):
            dp[0][j] = matrix[0][j]
            ans = max(ans, matrix[0][j])
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 1:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    ans = max(ans, dp[i][j])
        return ans ** 2
```

---

## LeetCode 33 - 二分查找

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left, right = 0, n - 1
        while left <= right:  # 注意要加等号！
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if target > nums[mid]:
                left = mid + 1
            else:
                right = mid - 1
        return -1
```

---

## LeetCode 33 - 搜索旋转排序数组

**原题描述：**
- 升序数组在某点旋转（无重复），在 O(log n) 内查找 target。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left, right = 0, n - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            # 左边有序
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # 右边有序
            else:
                if nums[right] >= target > nums[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1
```

---

## LeetCode 279 - 完全平方数

**原题描述：**
- 给定 n，求最少需要多少个完全平方数使其和等于 n。如 12=4+4+4，答案为 3。

```python
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [float("inf")] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                if j * j > i:
                    break
                dp[i] = min(dp[i], dp[i - j * j] + 1)
        return dp[-1]
```

---

## （非leetcode原题）找数组最小值和最大值（最小比较次数）

**问题：** 用最少比较次数同时找出数组的最小值和最大值。

**朴素：** 分别找最小、最大，共 2n-2 次比较。

**优化：配对比较法**

**详细思路：** 两两配对，先比较组内大小（1 次），较小者与全局 min 比、较大者与全局 max 比（2 次），每对共 3 次。n 偶数：3n/2-2；n 奇数：先取第一个为 min=max，剩余 (n-1)/2 对，3(n-1)/2 次。比朴素少约 25%。

```python
def find_min_max(arr: list) -> tuple:
    """配对比较法：约 1.5n 次比较"""
    n = len(arr)
    if n == 0:
        return None, None
    if n == 1:
        return arr[0], arr[0]
    # 初始化第一对
    if arr[0] < arr[1]:
        min_val, max_val = arr[0], arr[1]
    else:
        min_val, max_val = arr[1], arr[0]
    # 从第 2 个元素开始，每次处理两个
    i = 2
    while i < n - 1:
        if arr[i] < arr[i + 1]:
            lo, hi = arr[i], arr[i + 1]
        else:
            lo, hi = arr[i + 1], arr[i]
        min_val = min(min_val, lo)
        max_val = max(max_val, hi)
        i += 2
    if i < n:  # n 为奇数，剩余一个
        min_val = min(min_val, arr[i])
        max_val = max(max_val, arr[i])
    return min_val, max_val
```

---

## LeetCode 53 - 最大子数组和

**原题描述：**
- 求连续子数组的最大和。
- 示例：[-2,1,-3,4,-1,2,1,-5,4] → 6（[4,-1,2,1]）

动态规划：

假设 `nums` 数组的长度是 $n$，下标从 $0$ 到 $n-1$。

我们用 $f(i)$ 代表以第 $i$ 个数结尾的「连续子数组的最大和」，那么很显然我们要求的答案就是：

$$\max_{0 \leq i \leq n-1}\{f(i)\}$$

因此我们只要求出每个位置的 $f(i)$，然后返回 $f$ 数组中的最大值即可。那么我们如何求 $f(i)$ 呢？我们可以考虑 $nums[i]$ 单独成为一段还是加入 $f(i-1)$ 对应的那一段，这取决于 $nums[i]$ 和 $f(i-1)+nums[i]$ 的大小，我们希望获得一个比较大的，于是可以写出这样的动态规划转移方程：

$$f(i) = \max\{f(i-1) + nums[i], nums[i]\}$$

不难给出一个时间复杂度 $O(n)$、空间复杂度 $O(n)$ 的实现，即用一个 $f$ 数组来保存 $f(i)$ 的值，用一个循环求出所有 $f(i)$。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(nums[i], dp[i - 1] + nums[i])
        return max(dp)
```

考虑到 $f(i)$ 只和 $f(i-1)$ 相关，于是我们可以只用一个变量 $pre$ 来维护对于当前 $f(i)$ 的 $f(i-1)$ 的值是多少，从而让空间复杂度降低到 $O(1)$，这有点类似「滚动数组」的思想。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        ans = summation = nums[0]
        for i in range(1, n):
            summation = max(nums[i], summation + nums[i])
            ans = max(ans, summation)
        return ans
```

---

## LeetCode 152 - 乘积最大子数组

**原题描述：**
- 求乘积最大的连续子数组乘积。数组中可能有负数。

动态规划：

如果我们用 $f_{\max}(i)$ 来表示以第 $i$ 个元素结尾的乘积最大子数组的乘积，$a$ 表示输入参数 `nums`，那么根据「53. 最大子序和」的经验，我们很容易推导出这样的状态转移方程：

$$f_{\max}(i) = \max_{i=1}^{n}\{f(i-1) \times a_i, a_i\}$$

它表示以第 $i$ 个元素结尾的乘积最大子数组的乘积可以考虑 $a_i$ 加入前面的 $f_{\max}(i-1)$ 对应的一段，或者单独成为一段，这里两种情况下取最大值。求出所有的 $f_{\max}(i)$ 之后选取最大的一个作为答案。

**可是在这里，这样做是错误的。为什么呢？**

因为这里的定义并不满足「最优子结构」。具体地讲，如果 $a = \{5, 6, -3, 4, -3\}$，那么此时 $f_{\max}$ 对应的序列是 $\{5, 30, -3, 4, -3\}$，按照前面的算法我们可以得到答案为 $30$，即前两个数的乘积，而实际上答案应该是全体数字的乘积。我们来想一想问题出在哪里呢？问题出在最后一个 $-3$ 所对应的 $f_{\max}$ 的值既不是 $-3$，也不是 $4 \times -3$，而是 $5 \times 30 \times (-3) \times 4 \times (-3)$。所以我们得到了一个结论：**当前位置的最优解未必是由前一个位置的最优解转移得到的。**

我们可以根据正负性进行分类讨论。

考虑当前位置如果是一个负数的话，那么我们希望以它前一个位置结尾的某个段的积也是个负数，这样就可以负负得正，并且我们希望这个积尽可能「负得更多」，即尽可能小。如果当前位置是一个正数的话，我们更希望以它前一个位置结尾的某个段的积也是个正数，并且希望它尽可能地大。于是这里我们可以再维护一个 $f_{\min}(i)$，它表示以第 $i$ 个元素结尾的乘积最小子数组的乘积，那么我们可以得到这样的动态规划转移方程：

$$f_{\max}(i) = \max_{i=1}^{n}\{f_{\max}(i-1) \times a_i, f_{\min}(i-1) \times a_i, a_i\}$$

$$f_{\min}(i) = \min_{i=1}^{n}\{f_{\max}(i-1) \times a_i, f_{\min}(i-1) \times a_i, a_i\}$$

它代表第 $i$ 个元素结尾的乘积最大子数组的乘积 $f_{\max}(i)$，可以考虑把 $a_i$ 加入第 $i-1$ 个元素结尾的乘积最大或最小的子数组的乘积中，二者加上 $a_i$，三者取大，就是第 $i$ 个元素结尾的乘积最大子数组的乘积。第 $i$ 个元素结尾的乘积最小子数组的乘积 $f_{\min}(i)$ 同理。

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        ans = maxn = minn = nums[0]
        for i in range(1, n):
            maxn, minn = max(nums[i], maxn * nums[i], minn * nums[i]), min(nums[i], maxn * nums[i], minn * nums[i])
            ans = max(ans, maxn, minn)
        return ans
```

实现上，有一个注意点：

maxn, minn = max(nums[i], maxn * nums[i], minn * nums[i]), min

不能写成

maxn = max(nums[i], maxn * nums[i], minn * nums[i])
minn = min(nums[i], maxn * nums[i], minn * nums[i])

即不能拆开写，因为maxn和minn的更新用的应该都是久的maxn和minn的值，如果有先后顺序，先更新了maxn再更新minn，minn所用的maxn就不是旧的值了。

---

## LeetCode 238 - 除自身以外数组的乘积

**原题描述：**
- 返回数组 answer，其中 answer[i] = 除 nums[i] 外其余元素乘积。不能用除法，O(n) 时间，O(1) 额外空间（不含输出）。

**解法：前缀积 × 后缀积**

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        left = [1] * n
        for i in range(1, n):
            left[i] = nums[i - 1] * left[i - 1]
        right = [1] * n
        for i in range(n - 2, -1, -1):
            right[i] = nums[i + 1] * right[i + 1]
        ans = [0] * n 
        for i in range(n):
            ans[i] = left[i] * right[i]
        return ans
```

优化一下，可以先算right，然后left用一个变量随着ans的遍历维护最新的值即可：

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        right = [1] * n
        for i in range(n - 2, -1, -1):
            right[i] = nums[i + 1] * right[i + 1]
        ans = [0] * n 
        left = 1
        for i in range(n):
            ans[i] = left * right[i]
            left *= nums[i]
        return ans
```

---

# 其他题

## LeetCode 49 - 字母异位词分组

**原题描述：** 将字母异位词（字符相同、顺序不同）分到同一组。如 ["eat","tea","tan","ate","nat","bat"] → [["eat","tea","ate"],["tan","nat"],["bat"]]。

字母异位词，即两个词只要在字母的分布一致即可，一个方法是对每个词统计26个字母的数量，得到一个长度为26的向量，两个词的统计向量相同就是字母异位词。

再简单一点，可以对每个词进行重排序，把所有字母异位词标准化。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        def sort(s):
            return "".join(sorted(list(s)))
        
        d = defaultdict(list)
        for s in strs:
            key = sort(s)
            d[key].append(s)
        return list(d.values())
```

---

## LeetCode 438 - 找到字符串中所有字母异位词

**原题描述：** 在 s 中找所有 p 的字母异位词的起始下标。如 s="cbaebabacd", p="abc" → [0,6]。

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        def check(d1, d2):  # 用于判断两个d是不是相同
            for k in d1:
                if d1[k] != d2[k]:
                    return False
            for k in d2:
                if d1[k] != d2[k]:
                    return False
            return True
        
        # 获取p的d
        dp = defaultdict(int)
        for c in p:
            dp[c] += 1

        # 获取s[:len(p) - 1]的d作为初始值
        ds = defaultdict(int)
        for c in s[:len(p) - 1]:
            ds[c] += 1

        ans = []
        for i in range(len(p) - 1, len(s)):
            # 更新ds
            ds[s[i]] += 1
            if i - len(p) >= 0:
                ds[s[i - len(p)]] -= 1

            # 判断是否符合
            if check(dp, ds):
                ans.append(i - len(p) + 1)
        return ans
```

---

## LeetCode 76 - 最小覆盖子串

**原题描述：** 找 s 中涵盖 t 所有字符的最小子串（t 中每个字符在子串中的出现次数不少于 t 中的次数）。如 s="ADOBECODEBANC", t="ABC" → "BANC"。

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        def check(dt, ds):  # 检查此时的s是否包含t的所有字符
            for k in dt:
                if ds[k] < dt[k]:
                    return False
            return True

        # 获取dt
        dt = defaultdict(int)
        for c in t:
            dt[c] += 1
        
        ds = defaultdict(int)
        left, right = 0, 0
        
        ans_left = -1
        ans_len = float("inf")
        
        while right < len(s):
            ds[s[right]] += 1
            
            while check(dt, ds):
                if right - left + 1 < ans_len:
                    ans_len = right - left + 1
                    ans_left = left
                ds[s[left]] -= 1
                left += 1

            right += 1

        if ans_left != -1:
            return s[ans_left: ans_left + ans_len]
        return "" 
```

---

## LeetCode 209 - 长度最小的子数组

**原题描述：** 找和 ≥ target 的最短连续子数组。如 nums=[2,3,1,2,4,3], target=7 → 2（子数组 [4,3]）。

和LeetCode 76 - 最小覆盖子串类似的题，还更简单点：

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        min_len = float("inf")
        summation = 0
        left, right = 0, 0
        while right < len(nums):
            summation += nums[right]
            while summation >= target:
                min_len = min(min_len, right - left + 1)
                summation -= nums[left]
                left += 1
            right += 1
        if min_len != float("inf"):
            return min_len
        return 0
```

---

## LeetCode 34 - 在排序数组中查找元素的第一个和最后一个位置

**原题描述：** 在有序数组中找 target 的起始和结束位置，O(log n)。若不存在返回 [-1,-1]。

直观的思路肯定是从前往后遍历一遍。用两个变量记录第一次和最后一次遇见 `target` 的下标，但这个方法的时间复杂度为 $O(n)$，没有利用到数组升序排列的条件。

由于数组已经排序，因此整个数组是单调递增的，我们可以利用二分法来加速查找的过程。

考虑 `target` 开始和结束位置，其实我们要找的就是数组中「第一个等于 `target` 的位置」(记为 `leftIdx`) 和「第一个大于 `target` 的位置减一」(记为 `rightIdx`)。

二分查找中，寻找 `leftIdx` 即为在数组中寻找第一个大于等于 `target` 的下标，寻找 `rightIdx` 即为在数组中寻找第一个大于 `target` 的下标，然后将下标减一。

实际上，除了找「第一个大于 `target` 的位置减一」，也可仿照找左边界的方法，直接找最后一个等于target的位置。

最后，因为 `target` 可能不存在数组中，因此我们需要重新校验我们得到的两个下标 `leftIdx` 和 `rightIdx`，看是否符合条件，如果符合条件就返回 `[leftIdx, rightIdx]`，不符合就返回 `[-1, -1]`。

第一个大于 `target` 的位置减一：

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)

        # ========== 找左边界：第一个等于 target 的位置 ==========
        # 即：第一个 >= target 的位置
        left_idx = n  # 初始化为 n，表示不存在
        left, right = 0, n - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] >= target:
                right = mid - 1
                left_idx = mid
            else:
                left = mid + 1
        
        # 校验 left_idx 是否合法
        if not (0 <= left_idx < n and nums[left_idx] == target):
            return [-1, -1]  # target 不存在，直接返回

        # ========== 找右边界：第一个大于 target 的位置减一 ==========
        # 即：第一个 > target 的位置，然后减 1
        right_bound = n  # 第一个 > target 的位置，初始化为 n（表示不存在）
        left, right = 0, n - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] > target:      # 严格大于
                right = mid - 1
                right_bound = mid       # 记录候选位置
            else:                       # nums[mid] <= target，继续往右找
                left = mid + 1
        
        # 第一个 > target 的位置是 right_bound，减一就是最后一个 = target 的位置
        right_idx = right_bound - 1

        # 校验（其实 left_idx 合法时，right_idx 一般也合法，但保险起见）
        if right_idx >= left_idx and nums[right_idx] == target:
            return [left_idx, right_idx]
        else:
            return [-1, -1]
```

直接找最后一个等于target的位置：

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)

        # 找第一个等于 `target` 的位置
        left_idx = n
        left, right = 0, n - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] >= target:
                right = mid - 1
                left_idx = mid
            else:
                left = mid + 1
        
        # 需要检查一下，有可能nums[mid] 一直> target，没有等于过，那么就是target不存在
        if n > left_idx >= 0 and nums[left_idx] == target:
            pass
        else:
            left_idx = -1
        
        # 找第一个大于 `target` 的位置
        right_idx = n
        left, right = 0, n - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
                right_idx = mid

        if n > right_idx >= 0 and nums[right_idx] == target:
            pass
        else:
            right_idx = -1

        return left_idx, right_idx
```

---

## LeetCode 81 - 搜索旋转排序数组 II

**原题描述：** 在可能含重复元素的旋转排序数组中查找 target，返回是否存在。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        n = len(nums)
        left, right = 0, n - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True
            
            # 情况1：无法判断哪边有序
            if nums[left] == nums[mid] == nums[right]:
                left += 1
                right -= 1
            # 左边有序
            elif nums[left] <= nums[mid]:
                # 往左边搜
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # 右边有序
            else:
                # 往右边搜
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return False
```

这里要注意的是

```
# 左边有序
elif nums[left] <= nums[mid]:
```

这里，必须用<=，不能用<。而先判断左边还是右边都可以，也可以先判断右边：

```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        n = len(nums)
        left, right = 0, n - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True
            
            # 情况1：无法判断哪边有序
            if nums[left] == nums[mid] == nums[right]:
                left += 1
                right -= 1
            
            
            # 右边有序
            elif nums[mid] <= nums[right]:
                # 往右边搜
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            # 左边有序
            else:
                # 往左边搜
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            
        return False
```

---

## LeetCode 322 - 零钱兑换

**原题描述：** 凑成 amount 的最少硬币数，无法则返回 -1。如 coins=[1,2,5], amount=11 → 3。

**解法：** 完全背包求最小值。`dp[j] = min(dp[j], dp[j-coin]+1)`，dp[0]=0，其余初始为 inf。

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float("inf")] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for c in coins:
                if i - c >= 0 and dp[i - c] != float("inf"):
                    dp[i] = min(dp[i], dp[i - c] + 1)
        if dp[-1] != float("inf"):
            return dp[-1]
        return -1
```

如果对coin先做一个排序，还可以增加一个提早退出的逻辑，在coin特别多的情况下，可以提高效率：

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float("inf")] * (amount + 1)
        dp[0] = 0
        coins.sort()
        for i in range(1, amount + 1):
            for c in coins:
                if i - c >= 0:
                    if dp[i - c] != float("inf"):
                        dp[i] = min(dp[i], dp[i - c] + 1)
                else:
                    break
        if dp[-1] != float("inf"):
            return dp[-1]
        return -1
```

---

## LeetCode 518 - 零钱兑换 II

**原题描述：** 凑成 amount 的硬币组合数（每种硬币无限）。如 amount=5, coins=[1,2,5] → 4 种组合。

**解法：** 完全背包求组合数

这道题中，给定总金额 `amount` 和数组 `coins`，要求计算金额之和等于 `amount` 的硬币组合数。其中，`coins` 的每个元素可以选取多次，且不考虑选取元素的顺序，因此这道题需要计算的是**选取硬币的组合数**。

可以通过**动态规划**的方法计算可能的组合数。用 `dp[x]` 表示金额之和等于 `x` 的硬币组合数，目标是求 `dp[amount]`。

动态规划的边界是 `dp[0] = 1`。只有当不选取任何硬币时，金额之和才为 0，因此只有 1 种硬币组合。

对于面额为 `coin` 的硬币，当 `coin ≤ i ≤ amount` 时，如果存在一种硬币组合的金额之和等于 `i - coin`，则在该硬币组合中增加一个面额为 `coin` 的硬币，即可得到一种金额之和等于 `i` 的硬币组合。因此需要遍历 `coins`，对于其中的每一种面额的硬币，更新数组 `dp` 中的每个大于或等于该面额的元素的值。

由此可以得到动态规划的做法：

- 初始化 `dp[0] = 1`；
- 遍历 `coins`，对于其中的每个元素 `coin`，进行如下操作：
  - 遍历 `i` 从 `coin` 到 `amount`，将 `dp[i - coin]` 的值加到 `dp[i]`。
- 最终得到 `dp[amount]` 的值即为答案。

上述做法不会重复计算不同的排列。因为外层循环是遍历数组 `coins` 的值，内层循环是遍历不同的金额之和，在计算 `dp[i]` 的值时，可以确保金额之和等于 `i` 的硬币面额的顺序，由于顺序确定，因此不会重复计算不同的排列。

例如：`coins = [1, 2]`，对于 `dp[3]` 的计算，一定是先遍历硬币面额 1 后遍历硬币面额 2，只会出现以下 2 种组合：

$$
\begin{aligned}
3 &= 1 + 1 + 1 \\
3 &= 1 + 2
\end{aligned}
$$

硬币面额 2 不可能出现在硬币面额 1 之前，即不会重复计算 `3 = 2 + 1` 的情况。

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin] 
        return dp[-1]
```

---

## LeetCode 2224 - 转化时间需要的最少操作数

**原题描述：** 将时间从 current 转化为 correct，每次可将分钟数 +1、+5、+15 或 +60，求最少操作数。

```python
class Solution:
    def convertTime(self, current: str, correct: str) -> int:
        if current == correct:
            return 0

        # step 1: 计算时差
        # 全部转化成分钟
        def get_min(t):
            h, m = t.split(':')
            h, m = int(h), int(m)
            min = h * 60 + m
            return min

        current_min = get_min(current)
        correct_min = get_min(correct)
        
        diff = correct_min - current_min
        if diff < 0:
            diff += 24 * 60
        
        # step 2：计算操作次数
        # 直接从60开始操作
        count = 0
        for step in [60, 15, 5, 1]:
            while diff - step >= 0:
                diff -= step
                count += 1
                if diff == 0:
                    return count
```

---

## LeetCode 1552 - 两球之间的磁力

**原题描述：** n 个篮子位置（数组 position），放 m 个球，最大化两球间的最小距离。

对于此题我们需要先思考一个子问题：给定 $n$ 个空篮子，$m$ 个球放置的位置已经确定。那么「最小磁力」我们该如何计算？

不难得出「最小磁力」为这 $m$ 个球中相邻两球距离的最小值的结论。对于 $i < j < k$ 三个位置的球，最小磁力一定是 $j-i$ 和 $k-j$ 的较小值，而不是跨越了位置 $j$ 的 $i$ 和 $k$ 的差值 $k-i$。

明确了给定位置最小磁力的计算方法，回到本题，在本题中 $m$ 个球的位置是由我们决定的，只知道空篮子的位置，且题目希望通过排列 $m$ 个球的位置来「最大化最小磁力」。

我们假定最终的答案是 $ans$，即这个时候最小磁力为 $ans$，那么我们知道小于 $ans$ 的答案一定也合法。因为既然我们存在一种放置的方法使得相邻小球间距的最小值大于等于 $ans$，那么也一定大于 $[1, ans-1]$ 中的任意一个值，而大于 $ans$ 的均不合法，因此我们可以对答案进行二分查找。

假设我们在 $[left, right]$ 的区间查找。每次取 $mid$ 为 $left$ 和 $right$ 的平均值，进行如下操作：

- 如果当前的 $mid$ 合法，则令 $ans = mid$，并将区间缩小为 $[mid+1, right]$；
- 如果当前的 $mid$ 不合法，则将区间缩小为 $[left, mid-1]$。

最后剩下的问题是如何判断答案是否合法，即给定一个答案 $x$，是否存在一种放置方法使得相邻小球的间距最小值大于等于 $x$。这个问题其实很好解决，相邻小球的间距最小值大于等于 $x$，其实就等价于相邻小球的间距均大于等于 $x$。我们预先对给定的篮子的位置进行排序，那么从贪心的角度考虑，第一个小球放置的篮子一定是 $position$ 最小的篮子，即排序后的第一个篮子。那么为了满足上述条件，第二个小球放置的位置一定要大于等于 $position[0] + x$，接下来同理。因此我们从前往后扫 $position$ 数组，看在当前答案 $x$ 下我们最多能在篮子里放多少个小球，我们记这个数量为 $cnt$，如果 $cnt$ 大于等于 $m$，那么说明当前答案下我们的贪心策略能放下 $m$ 个小球且它们间距均大于等于 $x$，为合法的答案，否则不合法。

```python
class Solution:
    def maxDistance(self, position: List[int], m: int) -> int:
        def check(mid):
            pre = position[0]  # 第一个放在位置0
            count = 1
            for i in range(1, len(position)):
                if position[i] - pre >= mid:
                    pre = position[i]  # 放下新一个
                    count += 1
                    if count >= m:
                        return True
            return False

        position.sort()
        left = min([position[i] - position[i - 1] for i in range(1, len(position))])
        right = position[-1] - position[0]
        ans = left
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans
```

---

## LeetCode 23 - 合并 K 个升序链表

**原题描述：** 将 k 个有序链表合并为一个有序链表。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap = []
        for i, node in enumerate(lists):
            if node:
                heapq.heappush(heap, (node.val, i, node))  # 因为node本身无法没有实现比较的函数，因此用val作比较；而val可能相同，因此后面再加一个i，这样就保证不会用到node来作大小比较

        dummy = cur = ListNode()
        while heap:
            val, i, node = heapq.heappop(heap)
            cur.next = node
            cur = cur.next
            if node.next:
                heapq.heappush(heap, (node.next.val, i, node.next))
        return dummy.next
```

或者直接暴力做法：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # 1. 收集所有链表中的值
        values = []
        for head in lists:
            while head:
                values.append(head.val)
                head = head.next
        
        # 2. 排序
        values.sort()
        
        # 3. 根据排序结果构建新链表
        dummy = ListNode(0)
        cur = dummy
        for val in values:
            cur.next = ListNode(val)
            cur = cur.next
        
        return dummy.next
```

---

## LeetCode 20 - 有效的括号

**原题描述：** 判断括号字符串是否有效。如 "()[]{}" 有效，"(]" 无效。

```python
class Solution:
    def isValid(self, s: str) -> bool:
        d = {
            ")": "(",
            "]": "[",
            "}": "{",
        }
        q = []
        for c in s:
            if c in d.values():
                q.append(c)
            else:
                if not q:
                    return False
                pre = q.pop()
                if pre != d[c]:
                    return False
        return True if not q else False
```

---

## LeetCode 231 - 2 的幂

**原题描述：** 判断 n 是否为 2 的幂。如 1,2,4,8 是，3,6 不是。

转成2进制，如果是2的幂，那么所有bit里有且只有1个bit是1，其他都是0：

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n < 0:
            return False
        return sum([int(bit) for bit in bin(n)[2:]]) == 1
```

或者可以用递归方法：

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 1:
            return True
        if n < 1:
            return False
        return self.isPowerOfTwo(n / 2)
```

---

## LeetCode 17 - 电话号码的字母组合

**原题描述：** 给定数字串（2-9），每个数字对应若干字母，返回所有可能的字母组合。

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        d = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }
        q = deque([""])
        for digit in digits:
            n = len(q)
            for i in range(n):
                comb = q.popleft()
                for c in d[digit]:
                    q.append(comb + c)
        return list(q)
```

注意deque的初始化方式。

---

## LeetCode 39 - 组合总和

**原题描述：** 无重复数组，每个数可重复选用，求和为 target 的所有组合。

**解法：** 回溯。每次可从当前索引 i 开始选（可重复），所以递归时仍传 i；若不选当前数则传 i+1。

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = []

        def dfs(idx, combination, remain):
            if remain == 0:
                ans.append(combination[:])
                return
            if remain < 0 or idx >= len(candidates):
                return
            
            # 不加idx这个，那么就一定从idx+1开始，而不会再从idx开始
            dfs(idx + 1, combination, remain)
            # 加idx这个；本来有两种情况：从idx开始或者从idx+1开始，但是idx+1开始的已经被包含在idx开始里的第一种选择里，因此不用重复选
            dfs(idx, combination + [candidates[idx]], remain - candidates[idx])
            
        dfs(0, [], target)
        return ans
```

加上剪枝加速：

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        ans = []

        def dfs(idx, combination, remain):
            if remain == 0:
                ans.append(combination)
                return
            if remain < 0 or idx >= len(candidates):
                return
            
            # 排序之后，后面的值都太大，直接剪掉所有后续尝试
            if remain < candidates[idx]:
                return 

            # 不加idx这个，那么就一定从idx+1开始，而不会再从idx开始
            dfs(idx + 1, combination, remain)
            # 加idx这个；本来有两种情况：从idx开始或者从idx+1开始，但是idx+1开始的已经被包含在idx开始里的第一种选择里，因此不用重复选
            dfs(idx, combination + [candidates[idx]], remain - candidates[idx])
            
        dfs(0, [], target)
        return ans
```

另外的写法：

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()

        ans = []
        comb = []
        def backtrack(idx, remain):
            if remain == 0:
                ans.append(comb.copy())
                return 
            for i in range(idx, len(candidates)):
                if candidates[i] > remain:
                    break
                comb.append(candidates[i])
                backtrack(i, remain - candidates[i])
                comb.pop()
                
        backtrack(0, target)
        return ans
```

---

## LeetCode 40 - 组合总和 II

**原题描述：** 有重复数组，每个数只能用一次，求和为 target 的所有不重复组合。

仿照上一题的写法的话，只把idx改成idx+1，超时了：

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = []
        candidates.sort()

        def dfs(idx, comb, remain):
            if remain == 0:
                ans.append(tuple(comb))
                return 
            if remain < 0 or idx >= len(candidates):
                return 
            
            if candidates[idx] > remain:
                return 

            dfs(idx + 1, comb, remain)
            dfs(idx + 1, comb + [candidates[idx]], remain - candidates[idx])

        dfs(0, [], target)
        ans = list(set(ans))
        return ans
```

能通过的方法：

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        comb = []
        ans = []

        def dfs(idx, remain):
            if remain == 0:
                ans.append(comb[:])
                return 
            for i in range(idx, len(candidates)):
                if candidates[i] > remain:
                    break
                if i > idx and candidates[i] == candidates[i - 1]:
                    continue
                comb.append(candidates[i])
                dfs(i + 1, remain - candidates[i])
                comb.pop()

        dfs(0, target)
        return ans
```

为什么能保证没有重复？
含义：在同一层递归（即同一轮 for i in range(idx, len(candidates))）中，对于相同数值，只选第一个，后面的都直接跳过。
原因：在同一层里，选 candidates[i] 和选 candidates[i-1] 时，后面可选的子集是包含关系，得到的组合会重复。
举例：candidates = [1, 1, 2, 5]，target = 8：
选第一个 1：剩余 [1, 2, 5] 里找和为 7 → 得到 [1, 1, 2, 5] 等
选第二个 1：剩余 [2, 5] 里找和为 7 → 也会得到 [1, 1, 2, 5]
两个分支会产生相同组合，所以同层只选第一个，后面的用 continue 跳过即可。

---

## LeetCode 2 - 两数相加

**原题描述：** 两个链表表示逆序存储的非负整数，求和并以逆序链表返回。如 2->4->3 表示 342。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        addition = 0
        dummy = cur = ListNode()
        while l1 or l2:
            if not l1:
                summation = l2.val + addition
                l2 = l2.next
            elif not l2:
                summation = l1. val + addition
                l1 = l1.next
            else:
                summation = l1.val + l2.val + addition
                l1 = l1.next
                l2 = l2.next

            addition = summation // 10
            summation = summation % 10
            cur.next = ListNode(summation)
            cur = cur.next
        if addition:
            cur.next = ListNode(addition)
        return dummy.next
```

或者把addition也放到判断里，简化一下：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        addition = 0
        dummy = cur = ListNode()
        while l1 or l2 or addition:
            summation = addition
            if l2:
                summation += l2.val
                l2 = l2.next
            if l1:
                summation += l1. val
                l1 = l1.next

            addition = summation // 10
            summation = summation % 10
            cur.next = ListNode(summation)
            cur = cur.next
        
        return dummy.next
```

---

## LeetCode 25 - K 个一组翻转链表

**原题描述：** 每 k 个节点一组翻转链表，不足 k 的保留原样。如 k=2: 1->2->3->4->5 → 2->1->4->3->5。

暴力法，OOM了：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # 开辟一个数组，把每个节点存下来，并按k各个一组分组
        nodes = []
        p = head
        while p:
            nodes.append(p)
            p = p.next
        
        groups = []
        group = []
        for node in nodes:
            group.append(node)
            if len(group) == k:
                groups.append(group)
                group = []
        if group:
            groups.append(group)

        # 长度为k的组进行翻转
        for i in range(len(groups)):
            if len(groups[i]) == k:
                groups[i] = groups[i][::-1]

        # 重新拼接
        dummy = cur = ListNode()
        for group in groups:
            for node in group:
                cur.next = node
                cur = cur.next
        return dummy.next
```

另一种好点的暴力方法：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        p = head
        nodes = []
        h = dummy = ListNode()
        while p:
            if len(nodes) < k:
                nodes.append(p)
                p = p.next
            else:
                # 先翻转，清空
                for i in range(len(nodes) - 1, -1, -1):
                    h.next = nodes[i]
                    h = h.next
                nodes = []

                nodes.append(p)
                p = p.next
        if nodes:
            if len(nodes) < k:
                for node in nodes:
                    h.next = node
                    h = h.next
            else:
                for i in range(len(nodes) - 1, -1, -1):
                    if i == 0:
                        nodes[i].next = None
                    h.next = nodes[i]
                    h = h.next
        return dummy.next
```

空间O(1)的方法：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        def reverse(head, tail):  # tail是这组最后一个
            pre, cur = None, head
            while pre != tail:  # 本来是cur != tail.next，但是tail可能是None？因此用pre != tail比较安全
                tmp = cur.next
                cur.next = pre
                pre, cur = cur, tmp
            return pre, head
        
        dummy = ListNode(next=head)
        pre = dummy

        while head:
            # 查看剩余部分长度是否大于k
            tail = pre
            for i in range(k):
                tail = tail.next
                if not tail:
                    return dummy.next
            
            # 足够k个，并且这时tail在这组k个节点的最后一个，而head是这组k个第一个
            nex = tail.next
            head, tail = reverse(head, tail)
            pre.next = head
            tail.next = nex
            pre = tail
            head = tail.next
        
        return dummy.next
```

---

## LeetCode 624 - 数组列表中的最大距离

**原题描述：** 给定多个升序数组，从每个数组各取一个数，求最大值与最小值差的最大值。

**解法：** 遍历每个数组，用其最小值与全局 max 的差、其最大值与全局 min 的差来更新答案，同时用该数组的首尾更新全局 min、max。

```python
class Solution:
    def maxDistance(self, arrays: List[List[int]]) -> int:
        max_dist = 0
        min_value, max_value = arrays[0][0], arrays[0][-1]
        for i, array in enumerate(arrays[1:]): 
            max_dist = max(array[-1] - min_value, max_value - array[0], max_dist)
            min_value = min(array[0], min_value)
            max_value = max(array[-1], max_value)
        return max_dist
```

---

## LeetCode 1689 - 十-二进制数的最少数目

**原题描述：** 如果一个十进制数字不含任何前导零，且每一位上的数字不是 0 就是 1 ，那么该数字就是一个 十-二进制数 。例如，101 和 1100 都是 十-二进制数，而 112 和 3001 不是。

给你一个表示十进制整数的字符串 n ，返回和为 n 的 十-二进制数 的最少数目。

示例 1：

输入：n = "32"  
输出：3  
解释：10 + 11 + 11 = 32  

示例 2：

输入：n = "82734"  
输出：8  

示例 3：

输入：n = "27346209830709182346"  
输出：9  

```python
class Solution:
    def minPartitions(self, n: str) -> int:
        return max(max([int(num) for num in str(n)]), 1)
```

---  

## LeetCode 1536 - 排布二进制网格的最少交换次数

给你一个 n x n 的二进制网格 grid，每一次操作中，你可以选择网格的 相邻两行 进行交换。

一个符合要求的网格需要满足主对角线以上的格子全部都是 0 。

请你返回使网格满足要求的最少操作次数，如果无法使网格符合要求，请你返回 -1 。

主对角线指的是从 (1, 1) 到 (n, n) 的这些格子。

```python
class Solution:
    def minSwaps(self, grid: List[List[int]]) -> int:
        # 找到每行最后一个1的位置，这样就能确定每行后面有多少个0
        # 最后一个1的index=i，那么后面0的数量就是n - 1 - i
        n = len(grid)
        pos = [-1] * n
        for i in range(n):
            for j in range(n - 1, -1, -1):
                if grid[i][j] == 1:
                    pos[i] = j
                    break
        
        ans = 0
        for i in range(n):  # 对于每一行，要向下找到第一个能符合要求的行
            # 找能符合要求的行，标准是pos <= i
            k = -1
            for j in range(i, n):
                if pos[j] <= i:  # 如果某一行没有1，那么pos[j] = -1，一定满足要求，因此pos数组初始化的时候用-1
                    ans += j - i  # 可以算出用冒泡交换所需的次数（因为只能交换相邻行）
                    k = j  # 记下来，实际交换的时候要用
                    break  # 记得退出！
            # 交换
            if k != -1:
                for j in range(k, i, -1):
                    pos[j], pos[j - 1] = pos[j - 1], pos[j]  # 不用真的交换grid，只需交换对应的pos就可以
            else:
                return -1  # k = -1说明没有找到符合要求的行
            
        return ans
```

---  

## LeetCode 1545 - 找出第 N 个二进制字符串中的第 K 位

给你两个正整数 $n$ 和 $k$，二进制字符串 $S_n$ 的形成规则如下：

- $S_1 = \text{"0"}$
- 当 $i > 1$ 时，$S_i = S_{i-1} + \text{"1"} + \text{reverse}(\text{invert}(S_{i-1}))$

其中 `+` 表示串联操作，`reverse(x)` 返回反转 $x$ 后得到的字符串，而 `invert(x)` 则会翻转 $x$ 中的每一位（0 变为 1，而 1 变为 0）。

例如，符合上述描述的序列的前 4 个字符串依次是：

- $S_1 = \text{"0"}$
- $S_2 = \text{"011"}$
- $S_3 = \text{"0111001"}$
- $S_4 = \text{"011100110110001"}$

请你返回 $S_n$ 的第 $k$ 位字符，题目数据保证 $k$ 一定在 $S_n$ 长度范围以内。

示例 1

**输入**：$n = 3, k = 1$  
**输出**：$\text{"0"}$  
**解释**：$S_3$ 为 $\text{"0111001"}$，其第 1 位为 $\text{"0"}$。

```python
class Solution:
    def findKthBit(self, n: int, k: int) -> str:
        def getnum(n):
            d = {
                "0": "1",
                "1": "0"
            }

            def invert(s):
                return "".join([d[bit] for bit in s])
            
            def reverse(s):
                return s[::-1]
            
            pre = "0"
            for i in range(n):
                new = pre + "1" + reverse(invert(pre))
                pre = new
            return pre
        
        num = getnum(n)
        return num[k - 1]
```

可以稍微优化一下：如果k比较小，可以不用算到n：

```python
class Solution:
    def findKthBit(self, n: int, k: int) -> str:
        def getnum(n):
            d = {
                "0": "1",
                "1": "0"
            }

            def invert(s):
                return "".join([d[bit] for bit in s])
            
            def reverse(s):
                return s[::-1]
            
            pre = "0"
            for i in range(n):
                new = pre + "1" + reverse(invert(pre))
                pre = new
            return pre
        
        # 优化：如果k比较小，可以不用算那么多
        # s的增长规则：len(sn) = 2 * len(sn-1) + 1
        true_n = 1
        num_len = 1
        while num_len < k:
            num_len = num_len * 2 + 1
            true_n += 1

        num = getnum(true_n)
        return num[k - 1]
```

|s|表示s的长度，那么|sn| = 2|sn-1| + 1, 那么|sn| + 1 = 2(|sn-1| + 1), 那么{|sn| + 1}是一个首项为2，公比为2的等差数列。  

因此：|sn| = 2^n - 1, |sn-1| = 2^(n - 1) - 1, 说明sn的左半边是sn-1（共2^(n - 1) - 1个数字），正中间是第2^(n - 1)个数字，也就是1，右半边长度也是2^(n - 1) - 1个数字。

那么就可以判断k在那个位置，然后通过递归解决。

分类讨论：

- 如果 $k < 2^{n-1}$，那么第 $k$ 个字符位于 $S_n$ 的左半，问题变成 $S_{n-1}$ 的第 $k$ 个字符。这可以递归解决。
- 如果 $k > 2^{n-1}$，那么第 $k$ 个字符位于 $S_n$ 的右半，问题变成 $S_{n-1}$ 反转后的第 $k - 2^{n-1}$ 个字符，即反转前的第 $2^{n-1} - (k - 2^{n-1}) = 2^n - k$ 个字符（比如 $k = 2^n - 1$ 对应反转前的第 1 个字符）。这个字符再翻转，即为 $S_n$ 的第 $k$ 个字符。这也可以递归解决。

递归边界：

- 如果 $n = 1$，那么返回 $S_1$ 唯一的字符 0。
- 如果 $k = 2^{n-1}$，那么返回 $S_n$ 正中间的字符 1。

```python
class Solution:
    def findKthBit(self, n: int, k: int) -> str:
        if n == 1:
            return "0"
        
        mid = 1 << (n - 1)  # 中间点，即“1”
        if k == mid:
            return "1"
        elif k < mid:
            return self.findKthBit(n - 1, k)
        else:
            k = mid * 2 - k
            return "0" if self.findKthBit(n - 1, k) == "1" else "1"
```

---  

## LeetCode 516. 最长回文子序列

给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。

子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。

从“最长回文子串”借鉴过来的方法：

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        if n < 2:
            return len(s)

        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        for length in range(2, n + 1):
            for i in range(n):
                j = i + length - 1
                if j >= n:
                    break
                
                if s[i] != s[j]:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
                else:
                    if length <= 3:
                        dp[i][j] = length
                    else:
                        dp[i][j] = dp[i + 1][j - 1] + 2
        return dp[0][-1]
```

或者官方的写法：

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[0] * n for _ in range(n)]

        for i in range(n - 1, -1, -1):  # 因为计算dp[i][j]会用到dp[i + 1][j]，因此i要从大到小算
            dp[i][i] = 1
            for j in range(i + 1, n):  # 因为计算dp[i][j]会用到dp[i][j - 1]，因此j要从小到大算
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return dp[0][-1]
```

最长回文子串也可以参考这种写法，实现上更为简单一点。

## LeetCode 1582. 二进制矩阵中的特殊位置

给定一个 m x n 的二进制矩阵 mat，返回矩阵 mat 中特殊位置的数量。

如果位置 (i, j) 满足 mat[i][j] == 1 并且行 i 与列 j 中的所有其他元素都是 0（行和列的下标从 0 开始计数），那么它被称为 特殊 位置。

直接按定义算：

```python
class Solution:
    def numSpecial(self, mat: List[List[int]]) -> int:
        def check(i, j):
            row_sum = sum(mat[i])
            col_sum = sum(list(zip(*mat))[j])
            return row_sum == 1 and col_sum == 1
        
        m, n = len(mat), len(mat[0])
        count = 0
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1 and check(i, j):
                    count += 1
        return count
```

加上剪枝，已经有1存在的行和列不用再检查了：

```python
class Solution:
    def numSpecial(self, mat: List[List[int]]) -> int:
        def check(i, j):
            row_sum = sum(mat[i])
            col_sum = sum(list(zip(*mat))[j])
            return row_sum == 1 and col_sum == 1
        
        m, n = len(mat), len(mat[0])
        count = 0
        rows = set()
        cols = set()
        for i in range(m):
            if i in rows:
                continue
            for j in range(n):
                if j in cols:
                    continue
                if mat[i][j] == 1:
                    rows.add(i)
                    cols.add(j)
                    if check(i, j):
                        count += 1
        return count
```

按照特殊位置的定义，那么只有当当前位置为1，且当前行的和为1，当前列的和也为1是，这个位置是特殊位置。行和列的和可以提前计算。

```python
class Solution:
    def numSpecial(self, mat: List[List[int]]) -> int:
        m, n = len(mat), len(mat[0])
        rows_sum = [sum(row) for row in mat]
        cols_sum = [sum(col) for col in zip(*mat)]

        count = 0
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 1 and rows_sum[i] == 1 and cols_sum[j] == 1:
                    count += 1
        return count
```

---

## LeetCode 712. 两个字符串的最小ASCII删除和

给定两个字符串s1 和 s2，返回 使两个字符串相等所需删除字符的 ASCII 值的最小和 。

示例 1:

输入: s1 = "sea", s2 = "eat"
输出: 231
解释: 在 "sea" 中删除 "s" 并将 "s" 的值(115)加入总和。
在 "eat" 中删除 "t" 并将 116 加入总和。
结束时，两个字符串相等，115 + 116 = 231 就是符合条件的最小和。

```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + ord(s1[i - 1])
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + ord(s2[j - 1])
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + ord(s1[i - 1]), dp[i][j - 1] + ord(s2[j - 1]))
        return dp[-1][-1]
```

---

## LeetCode 1758. 生成交替二进制字符串的最少操作数

给你一个仅由字符 '0' 和 '1' 组成的字符串 s 。一步操作中，你可以将任一 '0' 变成 '1' ，或者将 '1' 变成 '0' 。

交替字符串 定义为：如果字符串中不存在相邻两个字符相等的情况，那么该字符串就是交替字符串。例如，字符串 "010" 是交替字符串，而字符串 "0100" 不是。

返回使 s 变成 交替字符串 所需的 最少 操作数。

```python
class Solution:
    def minOperations(self, s: str) -> int:
        # 计算和候选答案的差异数，即操作数
        def diff(s1, s2):
            return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])
        
        n = len(s)
        # 一个长度下，能成为交替字符的只有两个候选，分别构造出来，计算和s的位数差，即可得操作数
        group_n = (n // 2 + 1)
        s1 = "01" * group_n
        s1 = s1[:n]
        s2 = "10" * group_n
        s2 = s2[:n]
        return min(diff(s, s1), diff(s, s2))
```

更快捷的写法

```python
class Solution:
    def minOperations(self, s: str) -> int:
        # 更快捷的写法
        # cnt是其中一个候选操作数
        cnt = sum(int(c) != i % 2 for i, c in enumerate(s))
        # len(s) - cnt是另一个候选的操作数
        return min(cnt, len(s) - cnt)
```

---

## LeetCode 1784. 检查二进制字符串字段

给你一个二进制字符串 s ，该字符串 不含前导零 。

如果 s 包含 零个或一个由连续的 '1' 组成的字段 ，返回 true​​​ 。否则，返回 false 。

```python
class Solution:
    def checkOnesSegment(self, s: str) -> bool:
        # 由于没有前导0，因此第一位一定是1，也就是至少有一块连续的1，那么如果隔了0之后，后面还有1，则一定是false
        count = 0
        for i in range(1, len(s)):
            if s[i] != s[i - 1]:
                count += 1
                if count >= 2:
                    return False
        return True
```

更进一步简化：

```python
class Solution:
    def checkOnesSegment(self, s: str) -> bool:
        return "01" not in s
```

---

## LeetCode 1980. 找出不同的二进制字符串

给你一个字符串数组 nums ，该数组由 n 个 互不相同 的二进制字符串组成，且每个字符串长度都是 n 。请你找出并返回一个长度为 n 且 没有出现 在 nums 中的二进制字符串。如果存在多种答案，只需返回 任意一个 即可。  

示例 1：  
输入：nums = ["01","10"]  
输出："11"  
解释："11" 没有出现在 nums 中。"00" 也是正确答案。  

示例 2：  
输入：nums = ["00","01"]  
输出："11"  
解释："11" 没有出现在 nums 中。"10" 也是正确答案。  

示例 3：  
输入：nums = ["111","011","001"]  
输出："101"  
解释："101" 没有出现在 nums 中。"000"、"010"、"100"、"110" 也是正确答案。  

方法1：转成十进制处理，再转回二进制

```python
class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        n = len(nums)
        vals = {int(num, 2) for num in nums}  # 把二进制字符串转成十进制

        # 寻找不再vals中的数
        # 这里的写法比较巧妙，值得理解
        val = 0
        while val in vals:
            val += 1

        # 找到之后，转成二进制
        # 这里写得也比较巧妙
        res = "{:b}".format(val)
        return "0" * (n - len(res)) + res
```

---

## LeetCode 525. 连续数组

给定一个二进制数组 nums , 找到含有相同数量的 0 和 1 的最长连续子数组，并返回该子数组的长度。  

示例 1：  
输入：nums = [0,1]  
输出：2  
说明：[0, 1] 是具有相同数量 0 和 1 的最长连续子数组。  

示例 2：  
输入：nums = [0,1,0]  
输出：2  
说明：[0, 1] (或 [1, 0]) 是具有相同数量 0 和 1 的最长连续子数组。  

示例 3：  
输入：nums = [0,1,1,1,1,1,0,0,0]  
输出：6  
解释：[1,1,1,0,0,0] 是具有相同数量 0 和 1 的最长连续子数组。  

思路：前缀和

一句话思路：把 0 看成 -1，计算和为 0 的最长子数组。

这是怎么想出来的？

**前置知识：前缀和。**

设 0 在的 `nums` 中的个数前缀和数组为 $S_0$：如果 `nums[i] = 0` 则视作 1，否则视作 0，计算这个序列的前缀和数组。

同理，设 1 在的 `nums` 中的个数前缀和数组为 $S_1$。

子数组 $[l, r)$ 中的 1 和 0 的出现次数相等，即

$$S_1[r] - S_1[l] = S_0[r] - S_0[l]$$

移项得

$$S_1[r] - S_0[r] = S_1[l] - S_0[l]$$

定义数组 $sum[i] = S_1[i] - S_0[i]$，问题变成：

- 计算数组 `sum` 中的一对相等元素的最远距离。（注意子数组 $[l, r)$ 的长度是 $r - l$，无需加一。）

枚举右，维护左。维护 `sum[i]` 首次出现的下标，再次遇到 `sum[i]` 时，用 $i$ 减去 `sum[i]` 首次出现的下标，即为子数组长度，更新答案的最大值。

从 $sum[i] = S_1[i] - S_0[i]$ 这个定义可以看出来，如果把原数组中的 0 视作 -1，我们计算的就是和为 0 的最长子数组。

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        nums = [1 if num else -1 for num in nums]
        summations = list(accumulate(nums, initial=0))
        d = {}
        ans = 0
        for i in range(len(summations)):
            if summations[i] in d:
                ans = max(ans, i - d[summations[i]])
            else:
                d[summations[i]] = i
        return ans
```

---

## LeetCode 3129. 找出所有稳定的二进制数组 I

给你 3 个正整数 zero ，one 和 limit 。

一个 二进制数组 arr 如果满足以下条件，那么我们称它是 稳定的 ：

0 在 arr 中出现次数 恰好 为 zero 。
1 在 arr 中出现次数 恰好 为 one 。
arr 中每个长度超过 limit 的 子数组 都 同时 包含 0 和 1 。
请你返回 稳定 二进制数组的 总 数目。

由于答案可能很大，将它对 109 + 7 取余 后返回。

 

示例 1：

输入：zero = 1, one = 1, limit = 2

输出：2

解释：

两个稳定的二进制数组为 [1,0] 和 [0,1] ，两个数组都有一个 0 和一个 1 ，且没有子数组长度大于 2 。

示例 2：

输入：zero = 1, one = 2, limit = 1

输出：1

解释：

唯一稳定的二进制数组是 [1,0,1] 。

二进制数组 [1,1,0] 和 [0,1,1] 都有长度为 2 且元素全都相同的子数组，所以它们不稳定。

示例 3：

输入：zero = 3, one = 3, limit = 2

输出：14

解释：

所有稳定的二进制数组包括 [0,0,1,0,1,1] ，[0,0,1,1,0,1] ，[0,1,0,0,1,1] ，[0,1,0,1,0,1] ，[0,1,0,1,1,0] ，[0,1,1,0,0,1] ，[0,1,1,0,1,0] ，[1,0,0,1,0,1] ，[1,0,0,1,1,0] ，[1,0,1,0,0,1] ，[1,0,1,0,1,0] ，[1,0,1,1,0,0] ，[1,1,0,0,1,0] 和 [1,1,0,1,0,0] 。

```python
```

---

# 付费题

## LeetCode 1056 - 易混淆数

**原题描述：** 判断旋转 180° 后是否得到有效且不同的数。0→0,1→1,6→9,8→8,9→6，其他无效。

**解法：** 建立映射，将数字翻转并逐位转换，检查是否有效且与原数不同。

```python
def confusingNumber(n: int) -> bool:
    m = {0: 0, 1: 1, 6: 9, 8: 8, 9: 6}
    x, orig = 0, n
    while n:
        d = n % 10
        if d not in m:
            return False
        x = x * 10 + m[d]
        n //= 10
    return x != orig
```

---

## LeetCode 1427 - 字符串的左右移

**原题描述：** 给定字符数组 s 和数组 shift（每个元素为 [direction, amount]），direction 0 表示左移、1 表示右移，对 s 执行所有操作。

**解法：** 累计净位移（左移为负、右移为正），最后一次性执行。左移 k 位 = `s[k:] + s[:k]`；右移 k = 左移 n-k。

```python
def stringShift(s: str, shift: list) -> str:
    n = len(s)
    total = 0
    for d, a in shift:
        total += a if d == 1 else -a
    total %= n
    if total == 0:
        return s
    # 右移 total 位 = 左移 n-total 位
    k = (n - total) % n
    return s[k:] + s[:k]
```

---

## LeetCode 161 - 相隔为 1 的编辑距离

**原题描述：** 判断两字符串的编辑距离是否为 1（恰好一次插入、删除或替换）。

**解法：** 长度差 > 1 则 false。长度差为 1：长串删一个字符后应与短串相同。长度相同：只能有一处不同。

```python
def isOneEditDistance(s: str, t: str) -> bool:
    if abs(len(s) - len(t)) > 1:
        return False
    if len(s) > len(t):
        s, t = t, s
    for i in range(len(s)):
        if s[i] != t[i]:
            if len(s) == len(t):
                return s[i + 1:] == t[i + 1:]
            return s[i:] == t[i + 1:]
    return len(t) - len(s) == 1
```

---

## LeetCode 186 - 反转字符串中的单词 II

**原题描述：** 给定字符数组 s（单词间有空格），就地反转每个单词的顺序，但单词内字符顺序不变。如 "the sky is blue" → "blue is sky the"。

**解法：** 先整体反转，再逐个单词反转。

```python
def reverseWords(s: list) -> None:
    def reverse(i, j):
        while i < j:
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1

    reverse(0, len(s) - 1)
    left = 0
    for i in range(len(s) + 1):
        if i == len(s) or s[i] == ' ':
            reverse(left, i - 1)
            left = i + 1
```

---

## LeetCode 1055 - 形成字符串的最短路径

**原题描述：** 从 source 中选子序列形成 target，求最少需要多少个 source 的连续子串。每个子串是 source 的一个连续段，且按顺序拼接后得到 target。

**解法：** 贪心

**详细思路（新手向）：** 从左到右扫描 target，在 source 中找匹配。用指针记录 source 的当前位置。若 target 当前字符在 source 剩余部分能找到则继续；若找不到则必须新开一段，从 source 开头重新匹配，段数 +1。贪心每次尽可能长地在一段内匹配，能让段数最少。

```python
def shortestWay(source: str, target: str) -> int:
    t = 0
    res = 0
    while t < len(target):
        res += 1
        found = False
        for c in source:
            if t < len(target) and c == target[t]:
                t += 1
                found = True
        if not found:
            return -1
    return res
```

---

## LeetCode 159 - 至多包含两个不同字符的最长子串

**原题描述：** 找最多含 2 种字符的最长子串长度。如 "eceba" → 3（"ece"）。

**解法：** 滑动窗口

**详细思路（新手向）：**

1. **目标**：找最长的不含超过 2 种字符的子串。是 LeetCode 3（无重复字符）的扩展。

2. **滑动窗口**：右指针扩展，用哈希表记录窗口内每种字符的出现次数。当字符种类 > 2 时，左指针收缩，直到种类 ≤ 2。每次扩展后更新最大长度。

3. **收缩**：左指针右移时，对应字符计数减 1；若减到 0 则从哈希表删除，种类数减 1。

```python
from collections import defaultdict

def lengthOfLongestSubstringTwoDistinct(s: str) -> int:
    cnt = defaultdict(int)
    left = 0
    res = 0
    for right in range(len(s)):
        cnt[s[right]] += 1
        while len(cnt) > 2:
            cnt[s[left]] -= 1
            if cnt[s[left]] == 0:
                del cnt[s[left]]
            left += 1
        res = max(res, right - left + 1)
    return res
```

---

## LeetCode 249 - 移位字符串分组

**原题描述：** 若字符串可经过循环移位（a→b, b→c, ..., z→a）变为另一字符串，则同组。

---

## LeetCode 266 - 回文排列

**原题描述：** 判断字符串能否重排成回文。如 "code" 不能，"aab" 可以（"aba"）。

**解法：** 回文对称，最多一个字符出现奇数次，其余均为偶数次。

```python
from collections import Counter

def canPermutePalindrome(s: str) -> bool:
    cnt = Counter(s)
    odd = sum(1 for v in cnt.values() if v % 2 == 1)
    return odd <= 1
```

---

## LeetCode 280 - 摆动排序

**原题描述：** 就地重排数组使得 `nums[0] <= nums[1] >= nums[2] <= nums[3]...`（相邻大小交替）。

**解法：** 遍历，偶数位应小于等于相邻，奇数位应大于等于相邻。若不满足则与相邻交换。

```python
def wiggleSort(nums: list) -> None:
    for i in range(1, len(nums)):
        if (i % 2 == 1 and nums[i] < nums[i - 1]) or (i % 2 == 0 and nums[i] > nums[i - 1]):
            nums[i], nums[i - 1] = nums[i - 1], nums[i]
```

# 有点难

## LeetCode 76 - 最小覆盖子串

## LeetCode 239 - 滑动窗口最大值

## LeetCode 516 - 最长回文子序列

**原题描述：** 求最长回文子序列长度（子序列不要求连续）。如 "bbbab" → 4（"bbbb"）。

**解法：** 区间 DP

**详细思路（新手向）：**

1. **和回文子串的区别**：子序列可以不连续。如 "bbbab" 的最长回文子序列是 "bbbb"（长度为4），不是连续子串。

2. **状态**：`dp[i][j]` = 区间 s[i:j+1] 内的最长回文子序列长度。

3. **递推**：
   - 若 s[i] == s[j]：两端的字符可以一起选，`dp[i][j] = dp[i+1][j-1] + 2`
   - 否则：s[i] 和 s[j] 至少有一个不选。不选 s[i] 则 `dp[i+1][j]`，不选 s[j] 则 `dp[i][j-1]`，取较大值

4. **遍历顺序**：dp[i][j] 依赖 dp[i+1][j-1]、dp[i+1][j]、dp[i][j-1]，即左下、下、左。所以要按**区间长度**从小到大地填：先算长度 2 的区间，再算长度 3，以此类推。或者 i 从大到小、j 从小到大。

```python
def longestPalindromeSubseq(s: str) -> int:
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]
```

- **时间复杂度：** O(n²)
- **空间复杂度：** O(n²)

---

## LeetCode 4 - 寻找两个正序数组的中位数

**原题描述：**
- 给定两个正序数组 nums1、nums2，求合并后的中位数。要求 O(log(m+n))。

**解法：二分查找**

**详细思路（新手向）：**

1. **等价问题**：合并后求中位数，等价于把两个数组"切开"，左半部分的数量 = 右半部分（或差1），且左半的最大值 ≤ 右半的最小值。中位数就由左半最大和右半最小决定。

2. **切点**：在较短的数组 nums1 上二分切点 i。nums1 左半取 [0,i)，nums2 左半取 [0,j)，其中 j = (m+n+1)//2 - i，保证左半总数为一半（或一半多1）。这样左半最大值 = max(nums1[i-1], nums2[j-1])，右半最小值 = min(nums1[i], nums2[j])。

3. **二分条件**：需要满足 nums1[i-1] ≤ nums2[j] 且 nums2[j-1] ≤ nums1[i]，这样左半才都 ≤ 右半。若 nums2[j-1] > nums1[i]，说明 nums1 左半取少了，i 要增大；若 nums1[i-1] > nums2[j]，说明 nums1 左半取多了，i 要减小。

4. **边界**：i=0 或 j=0 时表示某数组左半为空；i=m 或 j=n 时表示某数组右半为空，需单独判断。

```python
def findMedianSortedArrays(nums1: list, nums2: list) -> float:
    """二分较短数组的切点 i，使左半最大值<=右半最小值"""
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    half = (m + n + 1) // 2
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2
        j = half - i
        if i < m and nums2[j - 1] > nums1[i]:
            lo = i + 1
        elif i > 0 and nums1[i - 1] > nums2[j]:
            hi = i - 1
        else:
            if i == 0:
                max_left = nums2[j - 1]
            elif j == 0:
                max_left = nums1[i - 1]
            else:
                max_left = max(nums1[i - 1], nums2[j - 1])
            if (m + n) % 2 == 1:
                return max_left
            if i == m:
                min_right = nums2[j]
            elif j == n:
                min_right = nums1[i]
            else:
                min_right = min(nums1[i], nums2[j])
            return (max_left + min_right) / 2.0
    return 0.0
```

- **时间复杂度：** O(log(min(m,n)))

---

## LeetCode 239 - 滑动窗口最大值

**原题描述：** 给定数组和窗口大小 k，求每个窗口的最大值。如 nums=[1,3,-1,-3,5,3,6,7], k=3 → [3,3,5,5,6,7]。

**解法：** 单调递减双端队列

**详细思路（新手向）：**

1. **暴力为何慢**：每个窗口扫一遍找最大，O(nk)。我们需要更快地知道"当前窗口的最大值"。

2. **单调队列思想**：维护一个**单调递减**的队列（队头最大），只存**有可能成为某个窗口最大值**的下标。若后面进来一个更大的数，那么前面比它小的数就不可能再成为最大值了（因为它们会先出窗口，且值更小），所以可以从队尾弹出。

3. **两个操作**：
   - **入队**：新元素入队前，从队尾弹出所有值比它小的（它们已无希望成为后续窗口的最大值），再入队
   - **出队**：若队头下标已经滑出窗口，则出队。队头始终是当前窗口最大值的下标

4. **举例**：窗口 [1,3,-1]，队列存下标 0,1,2，值为 1,3,-1。3 入队时弹出 1；-1 入队时 3>-1 不弹。队头是 1（值为3），即当前窗口最大值 3。

```python
from collections import deque

def maxSlidingWindow(nums: list, k: int) -> list:
    """单调递减队列：队头为当前窗口最大值"""
    q = deque()  # 存下标
    res = []
    for i in range(len(nums)):
        # 移除超出窗口的队头
        while q and q[0] <= i - k:
            q.popleft()
        # 从队尾移除比当前小的（它们不可能再成为最大值）
        while q and nums[q[-1]] < nums[i]:
            q.pop()
        q.append(i)
        if i >= k - 1:
            res.append(nums[q[0]])
    return res
```

- **时间复杂度：** O(n)

---

## LeetCode 983 - 最低票价

**原题描述：** 给定出行日期 days 和三种票价 costs（1/7/30 天），求覆盖所有出行日的最低花费。

**解法：** 动态规划

**详细思路（新手向）：**

1. **状态**：`dp[i]` = 覆盖第 1 天到第 i 天所有出行日的最小花费。

2. **转移**：若第 i 天不出行，`dp[i]=dp[i-1]`（继承前一天的花费）。若第 i 天出行，有三种选择：买 1 天票（dp[i-1]+costs[0]）、买 7 天票（dp[i-7]+costs[1]，若 i<7 则 dp[0]=0）、买 30 天票（dp[i-30]+costs[2]）。取三者最小值。

3. **注意**：dp 数组长度要到 max(days)，需要遍历 1 到 max(days) 的每一天，用集合记录出行日以便 O(1) 判断。

```python
def mincostTickets(days: list, costs: list) -> int:
    days_set = set(days)
    dp = [0] * (max(days) + 1)
    for i in range(1, len(dp)):
        if i not in days_set:
            dp[i] = dp[i - 1]
        else:
            dp[i] = min(
                dp[i - 1] + costs[0],
                dp[max(0, i - 7)] + costs[1],
                dp[max(0, i - 30)] + costs[2]
            )
    return dp[-1]
```

---

## LeetCode 2517 - 礼盒的最大甜蜜度

**原题描述：** 选 k 个不同价格的糖果，甜蜜度 = 所选糖果中任意两价格差的最小值。求最大甜蜜度。

**解法：** 二分答案 + 贪心验证

**详细思路（新手向）：**

1. **二分答案**：甜蜜度的取值范围是 [0, max(price)-min(price)]。我们二分一个值 x，判断"能否选出 k 个糖果使得甜蜜度至少为 x"。

2. **贪心验证**：若甜蜜度至少为 x，则任意两选中糖果的差价都 ≥ x。排序后，贪心从左往右选：第一个必选，之后每次选"与上一个选中糖果差价 ≥ x"且位置最靠左的。若最终能选满 k 个，则 x 可行。

3. **为何贪心对？** 要让选出的 k 个糖果两两差价都 ≥ x，等价于相邻选中糖果的差价 ≥ x（排序后）。贪心每次选满足条件且最靠左的，能给后续留出更多空间。

```python
def maximumTastiness(price: list, k: int) -> int:
    price.sort()
    lo, hi = 0, price[-1] - price[0]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cnt, last = 1, price[0]
        for p in price[1:]:
            if p - last >= mid:
                cnt += 1
                last = p
        if cnt >= k:
            lo = mid
        else:
            hi = mid - 1
    return lo
```

---  

# 排序算法

## 冒泡排序  

[经典排序算法](https://www.runoob.com/w3cnote/bubble-sort.html)

```python
def bubble_sort(arr):
    # 降序

    swapped = True  # 上一轮是否发生了交换，如果否，那么就是已经排序好了
    n = len(arr)
    while swapped:
        swapped = False
        for i in range(1, n):
            if arr[i - 1] > arr[i]:  # 决定排序是升还是降
                swapped = True
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
```

---

# 经验

## 堆  

堆是一种二叉树，其中每个上级节点的值都小于等于（大于等于）它的任意子节点。这一条件称为堆的不变性。  

基本概念：  
- https://docs.python.org/zh-cn/3.12/library/heapq.html  
- https://cloud.tencent.com/developer/article/2366201  
### python模块  

python模块：heapq，默认最小堆  

import的时候直接：import heapq  

python中可以用list来模拟（实现）堆  

重要函数：  
- heapq.heapify(x)：原地建堆，把x变成堆，复杂度nlogn  
- heapq.heappush(heap, item)，复杂度logn  
- heapq.heappop(heap)，复杂度logn  
- heapq.nlargest(n, iterable, key=None)，复杂度nlogn  
- heapq.nsmallest(n, iterable, key=None)，复杂度nlogn  

还有三个少用的：  
- heappushpop  
- heapreplace  
- merge  

python中只有最小堆，没有最大堆，对于数值型的元素，可以通过取反实现最大堆的效果  

例题：数组中的第K个最大元素  

### 自己实现堆

用list可以实现堆

在list中，父节点和子节点的关系：

（1）找子节点  

left节点下标 = 2 * i + 1  

right节点下标 = 2 * i + 2  

（2）左节点还是右节点  

首先需要知道一个节点是左节点还是右节点。list中，各个节点的排列是这样的：[根节点（可以当做右），左，右，左，右，...]  

因此index % 2 == 1的是左节点，index % 2 == 0的是右节点。  

（3）找父节点  

一个节点的parent为：  

parent = （i - 1）// 2  

具体需要实现push、pop、建堆等基本功能，有上浮之类的概念。  

实践中没用到，就不展开了，leetcode中会用heapq就行。  

## 队列，优先队列，双向队列  

1、队列  

python中的Queue，FIFO  

```
from queue import Queue  

Queue.qsize()：返回队列大小  
Queue.empty()：是否空
Queue.full()：是否满
Queue.get()：获取最先进queue的元素
Queue.put()：进queue

Queue.join()
Queue.task_nowait()
...
```

也有LIFO队列  

```
from queue import LifoQueue
```

2、优先队列  

带优先级的Queue（默认最小优先）  

```
from queue import PriorityQueue

插入的时候相比Queue多了一个优先级：
q.put((priority_num, data))

但是data也需要可以比较大小，如果data本身是不可比较的，那么可以自己实现一个比较大小的函数；比如data是一个链表节点，那么可以实现：

ListNode.__lt__ = lambda a, b: a.val < b.val

```

PriorityQueue可以用heap来实现  

3、双向队列：deque  

方法：pop()，popleft()，append()，appendleft()，extend，extendleft()  

```
from collection import deque

q = deque()  # 用法和list类似
q.appned()  # 用法和list类似
q.popleft()  # 用法和list类似

```

## 栈  

python的栈用list实现就行：append、pop

## 从quick sort到quick select  

快速排序的核心包括“哨兵划分” 和 “递归”：  
- 哨兵划分： 以数组某个元素（一般选取首元素）为基准数，将所有小于基准数的元素移动至其左边，大于基准数的元素移动至其右边。  
- 递归： 对 左子数组 和 右子数组 递归执行 哨兵划分，直至子数组长度为 1 时终止递归，即可完成对整个数组的排序。  

以leetcode 215. 数组中的第K个最大元素为例：  

```python
class Solution:
    def findKthLargest(self, nums, k):
        def quick_select(nums, k):
            # 随机选择基准数
            pivot = random.choice(nums)
            big, equal, small = [], [], []
            # 将大于、小于、等于 pivot 的元素划分至 big, small, equal 中
            for num in nums:
                if num > pivot:
                    big.append(num)
                elif num < pivot:
                    small.append(num)
                else:
                    equal.append(num)
            if k <= len(big):
                # 第 k 大元素在 big 中，递归划分
                return quick_select(big, k)
            if len(nums) - len(small) < k:
                # 第 k 大元素在 small 中，递归划分
                return quick_select(small, k - len(nums) + len(small))
            # 第 k 大元素在 equal 中，直接返回 pivot
            return pivot
        
        return quick_select(nums, k)
```

这题也可以用最小堆来实现  

## 二分  

python自带的二分查找，bisect库，一共就四个函数：  
- bisect.bisect_left  
- bisect.bisect_right  
- bisect.insort_left  
- bisect.insort_right  

底层是c++实现，python版本的实现可以参考：https://zhuanlan.zhihu.com/p/509628619  

标准二分法：  

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left, right = 0, n - 1
        while left <= right:  # 注意要加等号！
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if target > nums[mid]:
                left = mid + 1
            else:
                right = mid - 1
        return -1
```

例题：
- 704. 二分查找  
- 35. 搜索插入位置  
- 33. 搜索旋转排序数组  

## 回溯  

例题：  

- leetcode 22. 括号生成  
- leetcode 39. 组合总和  
- leetcode 40. 组合总和 II  
- leetcode 77. 组合  
- leetcode 46. 全排列
- leetcode 47. 全排列 II  

## 排列组合  

- leetcode 46. 全排列  

## 动态规划  

递归函数 + 边界条件  

例题：

- 最长回文子串  

## 其他  

### 背诵题  

#### 31. 下一个排列  

数学相关，有点巧妙，主要是想不到可以这样  

```
这道题不就这么个思路:

1.先倒序遍历数组, 找到第一个 nums[i] (前一个数比后一个数小的位置) (即nums[i] < nums[i+1]);
2.这个时候我们不能直接把后一个数nums[i+1] 跟前一个数nums[i]交换就完事了; 还应该从nums[i+1]-->数组末尾这一段的数据中 找出最优的那个值( 如何最优? 即比nums[i]稍微大那么一丢丢的数, 也就是 nums[i+1]-->数组末尾中, 比nums[i]大的数中最小的那个值)
3.找到之后, 跟num[i]交换, 这还不算是下一个排列, num[i]后面的数值还不够小, 所以还应当进升序排列
```

按这个逻辑确实可以过  

## 技巧

### 快速的矩阵转置方法

```python
mat_t = zip(*mat)  # 将mat转置的方法
```

#### zip解释

`zip` 是 Python 中一个非常强大且常用的内置函数，用来**将多个可迭代对象"压缩"在一起**。

基本用法：

```python
zip(*iterables)
```

把多个列表的**对应元素**配对打包成元组。

```python
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]

zipped = zip(names, ages)
print(list(zipped))
# [('Alice', 25), ('Bob', 30), ('Charlie', 35)]
```

就像拉链一样，把两排牙齿咬合在一起：

```
names:    Alice    Bob    Charlie
              ↓      ↓        ↓
ages:       25     30       35
              ↓      ↓        ↓
结果:    (Alice,25) (Bob,30) (Charlie,35)
```

关键点：

1. 返回的是迭代器

```python
z = zip([1,2], [3,4])
print(z)  # <zip object at 0x...>

# 需要转成 list 才能看内容
print(list(z))  # [(1, 3), (2, 4)]
```

2. 长度不一致时，以最短为准


```python
a = [1, 2, 3]
b = ['a', 'b']

list(zip(a, b))  # [(1, 'a'), (2, 'b')]  3 被忽略了
```

3. 可以解压（unzip）用 `zip(*)`

这是矩阵转置的核心技巧！

```python
pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
nums, letters = zip(*pairs)

print(nums)    # (1, 2, 3)
print(letters) # ('a', 'b', 'c')
```

`*pairs` 把列表解包成三个独立参数，相当于 `zip((1,'a'), (2,'b'), (3,'c'))`

常见用途：

| 场景 | 代码 |
|------|------|
| 同时遍历两个列表 | `for name, age in zip(names, ages):` |
| 创建字典 | `dict(zip(keys, values))` |
| 矩阵转置 | `zip(*matrix)` |
| 带索引的循环 | `for i, val in zip(range(len(lst)), lst):` |

### 进制转换

二进制字符串转十进制：在Python中，将二进制字符串转换为整数（int）的最简单方法是使用内置的 int() 函数，并将基数（base）参数设置为 2。  

```python
binary_str = "1101"
decimal_int = int(binary_str, 2)
print(decimal_int)  # 输出: 13
```

int转二进制字符串:  

```python
num = 10

# 1. 使用 bin() 函数 (结果为 '0b1010')
bin_str1 = bin(num) 
# 去除 '0b' 前缀: '1010'
bin_str1_clean = bin(num)[2:] 

# 2. 使用 format() 函数 (直接得到 '1010')
bin_str2 = format(num, 'b')

# 3. 使用 f-string (直接得到 '1010')
bin_str3 = f"{num:b}"

# 4. 补齐位数的二进制 (例如 8 位: '00001010')
bin_str4 = f"{num:08b}"

```

### accumulate 

参考[https://gairuo.com/p/python-itertools-accumulate](https://gairuo.com/p/python-itertools-accumulate)

itertools.accumulate() 是 Python 中 itertools 模块提供的一个函数，用于创建一个迭代器，该迭代器对输入的可迭代对象中的元素进行累积操作。默认情况下，累积操作是指将前一个元素与当前元素应用某个二元函数，并将结果作为下一个元素输出。  

语法为 itertools.accumulate(iterable[, func, *, initial=None])，与类似函数 functools.reduce() 机制差不多，但它不返回一个最终累积值，而是把累积的过程结果形成一个可迭代对象，可以理解这个序列里的内容是计算过程。  

大致相当于：  

```python
def accumulate(iterable, func=operator.add, *, initial=None):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], initial=100) --> 100 101 103 106 110 115
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = initial
    if initial is None:
        try:
            total = next(it)
        except StopIteration:
            return
    yield total
    for element in it:
        total = func(total, element)
        yield total
```

例如：  

```python
>>> data = [3, 4, 6, 2, 1, 9, 0, 7, 5, 8]
>>> list(accumulate(data, operator.mul))     # running product
[3, 12, 72, 144, 144, 1296, 0, 0, 0, 0]
>>> list(accumulate(data, max))              # running maximum
[3, 4, 6, 6, 6, 9, 9, 9, 9, 9]

# Amortize a 5% loan of 1000 with 4 annual payments of 90
>>> cashflows = [1000, -90, -90, -90, -90]
>>> list(accumulate(cashflows, lambda bal, pmt: bal*1.05 + pmt))
[1000, 960.0, 918.0, 873.9000000000001, 827.5950000000001]

# Chaotic recurrence relation https://en.wikipedia.org/wiki/Logistic_map
>>> logistic_map = lambda x, _:  r * x * (1 - x)
>>> r = 3.8
>>> x0 = 0.4
>>> inputs = repeat(x0, 36)     # 仅使用初始值
>>> [format(x, '.2f') for x in accumulate(inputs, logistic_map)]
['0.40', '0.91', '0.30', '0.81', '0.60', '0.92', '0.29', '0.79', '0.63',
 '0.88', '0.39', '0.90', '0.33', '0.84', '0.52', '0.95', '0.18', '0.57',
 '0.93', '0.25', '0.71', '0.79', '0.63', '0.88', '0.39', '0.91', '0.32',
 '0.83', '0.54', '0.95', '0.20', '0.60', '0.91', '0.30', '0.80', '0.60']
```

## 概念  

### 前缀树，Trie树，字典树  

[OI Wiki](https://oi-wiki.org/string/trie/#__tabbed_1_2)  

