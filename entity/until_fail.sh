#!/bin/bash
$@
while [[ $? -eq 0 ]]; do
    $@
done
