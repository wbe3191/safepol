#!/usr/bin/env bash
# Deploy Online Boutique v0.10.1 (unmodified upstream manifest) plus the benchmark's drift service
# "promoservice" into namespace "boutique".  Images must already be present locally (pull_images.sh).
set -eu
cd "$(dirname "$0")/../.."
kubectl create namespace boutique --dry-run=client -o yaml | kubectl apply -f -
kubectl -n boutique apply -f k8s/boutique.yaml
cat <<'EOF' | kubectl -n boutique apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: promoservice
  labels: {app: promoservice}
spec:
  replicas: 1
  selector: {matchLabels: {app: promoservice}}
  template:
    metadata: {labels: {app: promoservice}}
    spec:
      containers:
      - name: server
        image: busybox:latest
        imagePullPolicy: IfNotPresent
        command: ["sh", "-c", "mkdir -p /www && echo promo > /www/index.html && exec httpd -f -p 9000 -h /www"]
        ports: [{containerPort: 9000}]
        securityContext: {allowPrivilegeEscalation: false, readOnlyRootFilesystem: false}
---
apiVersion: v1
kind: Service
metadata:
  name: promoservice
  labels: {app: promoservice}
spec:
  selector: {app: promoservice}
  ports: [{name: promo, port: 9000, targetPort: 9000}]
EOF
# a probe pod used by the experiments to generate legitimate/attacker test flows (curl only)
cat <<'EOF' | kubectl -n boutique apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: probe
  labels: {app: probe}
spec:
  replicas: 1
  selector: {matchLabels: {app: probe}}
  template:
    metadata: {labels: {app: probe}}
    spec:
      containers:
      - name: curl
        image: curlimages/curl:8.10.1
        imagePullPolicy: IfNotPresent
        command: ["sleep", "infinity"]
EOF
kubectl -n boutique rollout status deploy --timeout=300s 2>&1 | tail -3
kubectl -n boutique get pods -o wide
