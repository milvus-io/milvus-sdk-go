# Changelog     

## milvus-sdk-go 0.4.5 

### Bug
---

### Improvement
---

### Feature
---
- \#82 - Change LoadCollection interface

### Task
---

## milvus-sdk-go 0.4.2 (2020-06-13)

### Bug
---
- \#37 - Fix sever version check for milvus-0.10.0
- \#39 - ListPartitions returns wrong result
### Improvement
---

### Feature
---

### Task
---

## milvus-sdk-go 0.4.1 (2020-05-29)

### Bug
---
- \#35 - Fix SeverVersion check for milvus-0.9.1

### Improvement
---

### Feature
---

### Task
---

## milvus-sdk-go 0.4.0 (2020-05-15)

### Bug
---

### Improvement
---
- \#26 - Change GetEntityByID to GetEntitiesByID
- \#29 - Filtering id = -1 result when total count < topk

### Feature
---
- \#25 - Add SearchByID interface
- \#31 - Rename sdk interfaces

### Task
---
- \#33 - Remove SearchByID

## milvus-sdk-go 0.3.0 (2020-04-15)

### Bug
---
- \#24 - sdk version error

### Improvement
---

### Feature
---

### Task
---

## milvus-sdk-go 0.2.0 (2020-03-31)

### Bug
---
- \#5 - Search interface bug
- \#10 - When the collection is empty, the search client throws exceptions
- \#15 - Search won't return status when return nq = 0
- \#20 - Receive large dataset failed
- \#21 - go sdk can not return error message if grpc failed

### Improvement
---
- \#9 - The INSERT method should return IDs
- \#18 - Change table to collection in proto

### Feature
---
- \#6 - Add substructure and superstructure metric types
- \#12 - Add annoy index type

### Task
---

## milvus-sdk-go 0.1.0 (2020-03-11)