# Contributing to Milvus Go SDK

Contributions of all kinds and from everyone are welcome. Simply file an issue stating your reason and plans for making the change, update CHANGELOG.md, and create a pull request to the current active branch. Make sure to mention the issue you filed in your PR description. Cheers!


## What contributions can I make?

Any contributions are allowed without changing the project architecture and interfaces here.
You are more than welcome to make contributions.


## How can I contribute?

### Development environment

You are recommended to develop using [Go](https://golang.org/dl/). We recommended using version 1.15 in this project.


### Coding Style

Please follow the [Milvus Contribution Guide](https://github.com/milvus-io/milvus/blob/master/CONTRIBUTING.md#coding-style).


## Run unit test with code coverage

Before submitting your PR, make sure you have the run unit test, and your code coverage rate is >= 80%.

```shell 
$ ./scripts/run_go_unittest.sh
```

You may need a Milvus server which is running when you run unit test. See more details on [Milvus server](https://github.com/milvus-io/milvus).


## Update CHANGLOG.md

Add issue tips into CHANGLOG.md, make sure all issue tips are sorted by issue number in ascending order.
