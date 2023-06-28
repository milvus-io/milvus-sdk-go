# Examples

here is some list to demonstrate the usage of `milvus-sdk-go`

- [Basic Usage](basic/basic.go) Shows some basic DDL(data definition language) like operations. Create collection, create partitions...
- [Insert and Search](insert/insert.go) Insert & search example, parses [films.csv](films.csv) and insert into collection and do searching
- [Index building](index/index.go) Index related creation/search example, time consumption compared as well
- [Calculate Distance](calcdistance/calc_distance.go) Calculate distance between vectors. Both ids or raw vectors example are presented.
- [Hello Milvus](hello_milvus/hello_milvus.go) Golang version of [hello_milvus](https://milvus.io/docs/v2.0.x/example_code.md)
- [Use database](database/database.go) Create, use and drop database of Milvus, isolate your data in the unique Milvus cluster.
