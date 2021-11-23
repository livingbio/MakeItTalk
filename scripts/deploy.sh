#!/bin/bash
# basic reference for writing script for travis

set -ev

# config k8s
g8 app deploy app.yaml --cluster=$CLUSTER --image-url=$IMAGE:$TAG --version=$TAG --backend=gke --verbose