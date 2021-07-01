#!/bin/bash

currentVersion="$(protoc --version)"
parts=($currentVersion)

if [ "${parts[0]}" != "libprotoc" ]; then
    echo "no libprotoc found"
    exit 1
fi
currentVersion=${parts[1]}
requiredVersion="3.0.0"

if [ "$(printf '%s\n%s' "$currentVersion" "$requiredVersion" | sort -V| head -n1)" = "$requiredVersion" ]; then
    echo "protoc version $currentVersion, ok"
    exit 0
else
    echo "protoc version needs to be greater than $requiredVersion, but $currentVersion found"
    exit 1
fi


