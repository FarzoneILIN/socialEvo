# 数独前置检查轨迹示例

保留 1 个场景配置的全部已有记录，8 个账本位中有 8 条模型回复。数独仅有前置记录，正式通信主对照没有回复。

以下展示消息及首个相关条件的原始回复，方便先读；完整提示、全部重复、评分和缺失状态见同目录 JSONL。

## message

场景 `sudoku_01`；记录 `sudoku_01/message/A`；模型 `gpt-5.4`。

```text
{"message":"My private givens: r2c5=3, r3c8=3, r4c8=9, r5c3=9, r6c4=9, r6c5=4, r7c4=4, r8c8=1, r8c9=3. From public+these I can deduce: r8c4=8, r9c4=6, r1c3=2, r2c3=5, r1c5=6, r1c8=4, r2c8=7, r5c8=6, r5c1=5, r1c1=3, r5c2=4, r6c2=3, r5c7=3, r7c3=1, r8c3=4, r3c6=4, r2c1=4. Also row 4 has c3/c5 = {7,8}, so target r4c5 is currently {7,8}. Current candidates I have: r1c6{8,9}, r1c7{1,8,9}, r1c9{1,8,9}; r2c2{8,9}, r2c7{8,9}; r3c2{8,9}, r3c5{8,9}; r4c3{7,8}, r4c5{7,8}; r5c6{7,8}, r5c9{7,8}; r6c3{7,8}, r6c7{1,7,8}, r6c9{1,7,8}; r7c1{7,9}, r7c7{7,9}; r8c1{7,9}, r8c5{7,9}; r9c6{7,9}, r9c9{7,9}."}
```

## baseline

场景 `sudoku_01`；记录 `sudoku_01/decision/full_information/A/r0`；模型 `gpt-5.4`。

```text
{"digit":7,"explanation":"Row 4 is 2,1,8,3,_,6,5,9,4, so the only missing digit is 7. Therefore r4c5=7."}
```

每个条件是独立对照分支，以上展示顺序不是连续对话或并行调用的全局时间线。
