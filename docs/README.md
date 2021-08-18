# API Documents

Below list the APIs that `Client` provides:

- [New Client](new_client.md): connect to a provided Milvus instance
- [List Collections](list_collections.md): list collections of the connected Milvus instance
- [Create Collection](create_collection.md): create collection according to the provided schema
- [Describe Collection](describe_collection.md): describe the collection with the specified name
- [Drop Collection](drop_collection.md): drop the collection with specified name
- [Load Collection](load_collection.md): load the collection for search by name
- [Release Collection](release_collection.md): release the loaded collection by name

- [Create Paritition](create_partition.md): create partition in a collection
- [Drop Partition](drop_partition.md): drop partition from a collection
- [Has Partition](has_partition.md): check if specified partition exists in the collection
- [Show Partitions](show_partitions.md): list the existing partitions of a specified collection
- [Load Partitions](load_partitions.md): load partitions of a collection for search
- [Release Partitions](release_partitions.md): release loaded partitions of a collection

- [Create Index](create_index.md): create index on the vector field of a collection
- [Drop Index](drop_index.md): drop index from the vector field of a collection
- [Describe index](describe_index.md): describe the index on the specified vector field
- [Get Index Build Progress](get_index_build_progress.md): get the index building progress information

- [Insert](insert.md): insert data into collection
- [Flush](flush.md): flush the inserted data
- [Search](search.md): search in the collection with provided criterion and vectors
