#!/usr/bin/env bash

set -e

echo "" > coverage.project.out

for d in $(go list ./... | grep -v vendor/examples/tests); do
    if [[ "$d" == *examples*  ]]; then
        continue
    fi
    if [[ "$d" == *test* ]]; then
        continue
    fi

    echo $d
    go test -race -coverprofile=coverage.out -covermode=atomic "$d" -v
    if [[ -f coverage.out ]]; then
        cat coverage.out >> coverage.project.out
        rm coverage.out
    fi
done

