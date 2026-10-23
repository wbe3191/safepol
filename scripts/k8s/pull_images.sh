#!/usr/bin/env bash
# Pull Online Boutique images through reachable mirrors and retag them to the names used in the manifest,
# so that the unmodified upstream manifest works on a host that cannot reach gcr.io / docker.io directly.
set -u
cd "$(dirname "$0")/../.."
GCR=gcr.m.daocloud.io
HUB=docker.m.daocloud.io
for svc in adservice cartservice checkoutservice currencyservice emailservice frontend loadgenerator paymentservice productcatalogservice recommendationservice shippingservice; do
  src=$GCR/google-samples/microservices-demo/$svc:v0.10.1
  dst=gcr.io/google-samples/microservices-demo/$svc:v0.10.1
  docker image inspect "$dst" >/dev/null 2>&1 && { echo "have $svc"; continue; }
  echo "pull $svc"; docker pull -q "$src" >/dev/null && docker tag "$src" "$dst" || echo "FAIL $svc"
done
for img in redis:alpine busybox:latest curlimages/curl:8.10.1 nicolaka/netshoot:v0.13; do
  docker image inspect "$img" >/dev/null 2>&1 && { echo "have $img"; continue; }
  echo "pull $img"; docker pull -q "$HUB/$img" >/dev/null && docker tag "$HUB/$img" "$img" || echo "FAIL $img"
done
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E 'microservices-demo|redis|busybox|curl|netshoot'
