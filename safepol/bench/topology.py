"""Benchmark topology: Google Online Boutique (11 microservices + redis) with external endpoints.

Dependency graph follows the upstream architecture diagram
(https://github.com/GoogleCloudPlatform/microservices-demo)."""
from __future__ import annotations

from typing import Dict, List, Tuple

from safepol.model.policy_model import (TCP, UDP, ExternalEndpoint, PortSpace, Topology, Workload)

NAMED_PORTS: Dict[str, Tuple[int, int]] = {
    "http-8080": (TCP, 8080),
    "cart-7070": (TCP, 7070),
    "catalog-3550": (TCP, 3550),
    "currency-7000": (TCP, 7000),
    "grpc-50051": (TCP, 50051),
    "checkout-5050": (TCP, 5050),
    "ad-9555": (TCP, 9555),
    "redis-6379": (TCP, 6379),
    "meta-80": (TCP, 80),
    "https-443": (TCP, 443),
    "mining-3333": (TCP, 3333),
    "mining-14444": (TCP, 14444),
    "revshell-4444": (TCP, 4444),
    "ssh-22": (TCP, 22),
    "dns-53": (UDP, 53),
    "promo-9000": (TCP, 9000),
}

SERVICES: List[Tuple[str, Dict[str, Tuple[int, int]], str, float]] = [
    # name, served ports, tier, image vulnerability score (CVSS-like 0-1, from a scanner)
    ("frontend", {"http-8080": NAMED_PORTS["http-8080"]}, "ingress", 0.35),
    ("cartservice", {"cart-7070": NAMED_PORTS["cart-7070"]}, "app", 0.20),
    ("productcatalogservice", {"catalog-3550": NAMED_PORTS["catalog-3550"]}, "app", 0.15),
    ("currencyservice", {"currency-7000": NAMED_PORTS["currency-7000"]}, "app", 0.45),
    ("paymentservice", {"grpc-50051": NAMED_PORTS["grpc-50051"]}, "app", 0.40),
    ("shippingservice", {"grpc-50051": NAMED_PORTS["grpc-50051"]}, "app", 0.10),
    ("emailservice", {"http-8080": NAMED_PORTS["http-8080"]}, "app", 0.30),
    ("checkoutservice", {"checkout-5050": NAMED_PORTS["checkout-5050"]}, "app", 0.20),
    ("recommendationservice", {"http-8080": NAMED_PORTS["http-8080"]}, "app", 0.50),
    ("adservice", {"ad-9555": NAMED_PORTS["ad-9555"]}, "app", 0.60),
    ("redis-cart", {"redis-6379": NAMED_PORTS["redis-6379"]}, "db", 0.25),
    ("loadgenerator", {}, "app", 0.30),
]

DEP_EDGES: List[Tuple[str, str, str]] = [
    ("Internet", "frontend", "http-8080"),
    ("loadgenerator", "frontend", "http-8080"),
    ("frontend", "productcatalogservice", "catalog-3550"),
    ("frontend", "currencyservice", "currency-7000"),
    ("frontend", "cartservice", "cart-7070"),
    ("frontend", "recommendationservice", "http-8080"),
    ("frontend", "shippingservice", "grpc-50051"),
    ("frontend", "checkoutservice", "checkout-5050"),
    ("frontend", "adservice", "ad-9555"),
    ("checkoutservice", "productcatalogservice", "catalog-3550"),
    ("checkoutservice", "shippingservice", "grpc-50051"),
    ("checkoutservice", "paymentservice", "grpc-50051"),
    ("checkoutservice", "emailservice", "http-8080"),
    ("checkoutservice", "currencyservice", "currency-7000"),
    ("checkoutservice", "cartservice", "cart-7070"),
    ("recommendationservice", "productcatalogservice", "catalog-3550"),
    ("cartservice", "redis-cart", "redis-6379"),
]

# Base legitimate request rate (connections per 5 s decision period) per dependency edge.
EDGE_RATE: Dict[Tuple[str, str], float] = {
    ("Internet", "frontend"): 60, ("loadgenerator", "frontend"): 40,
    ("frontend", "productcatalogservice"): 120, ("frontend", "currencyservice"): 90,
    ("frontend", "cartservice"): 70, ("frontend", "recommendationservice"): 60,
    ("frontend", "shippingservice"): 30, ("frontend", "checkoutservice"): 15,
    ("frontend", "adservice"): 50, ("checkoutservice", "productcatalogservice"): 15,
    ("checkoutservice", "shippingservice"): 15, ("checkoutservice", "paymentservice"): 15,
    ("checkoutservice", "emailservice"): 15, ("checkoutservice", "currencyservice"): 15,
    ("checkoutservice", "cartservice"): 15, ("recommendationservice", "productcatalogservice"): 60,
    ("cartservice", "redis-cart"): 80,
}

EXTERNALS = [
    ExternalEndpoint("Internet", "0.0.0.0/0", "Internet"),
    ExternalEndpoint("NodeCIDR", "10.0.0.0/16", "NodeCIDR"),
    ExternalEndpoint("MetadataService", "169.254.169.254/32", "MetadataService"),
    ExternalEndpoint("APIServer", "10.96.0.1/32", "APIServer"),
]


def make_port_space() -> PortSpace:
    return PortSpace(NAMED_PORTS)


def online_boutique(port_space: PortSpace | None = None, with_promo: bool = False) -> Topology:
    ps = port_space or make_port_space()
    wl = [Workload(name, "boutique", {"app": name}, ports, tier, tier != "untrusted", vuln)
          for name, ports, tier, vuln in SERVICES]
    edges = list(DEP_EDGES)
    if with_promo:
        # topology drift: a new service "promoservice" called by frontend and calling catalog
        wl.append(Workload("promoservice", "boutique", {"app": "promoservice"},
                           {"promo-9000": NAMED_PORTS["promo-9000"]}, "app", True, 0.55))
        edges += [("frontend", "promoservice", "promo-9000"),
                  ("promoservice", "productcatalogservice", "catalog-3550")]
    return Topology(wl, EXTERNALS, edges, ps)


def edge_rates(topo: Topology) -> Dict[Tuple[int, int], float]:
    rates = {}
    for (s, d, m) in topo.dep_edges:
        key = (topo.names[s], topo.names[d])
        rates[(s, d)] = EDGE_RATE.get(key, 30.0)
    return rates


def synthetic_topology(n: int, seed: int = 0, fanout: float = 2.5) -> Topology:
    """Scale-free-ish microservice call graph with n workloads (for scalability experiments).
    Edge structure mimics Alibaba microservice traces: few hubs, many leaves."""
    import numpy as np
    rng = np.random.default_rng(seed)
    ps = make_port_space()
    port_names = ["http-8080", "cart-7070", "catalog-3550", "currency-7000", "grpc-50051", "checkout-5050", "ad-9555"]
    wl, edges = [], []
    for i in range(n):
        pn = port_names[i % len(port_names)]
        tier = "db" if i % 20 == 19 else ("ingress" if i % 50 == 0 else "app")
        if tier == "db":
            pn = "redis-6379"
        wl.append(Workload(f"svc{i}", f"ns{i % 10}", {"app": f"svc{i}"}, {pn: NAMED_PORTS[pn]}, tier, True, float(rng.random())))
    # preferential attachment
    deg = np.ones(n)
    for i in range(1, n):
        k = max(1, int(rng.poisson(fanout)))
        probs = deg[:i] / deg[:i].sum()
        targets = rng.choice(i, size=min(k, i), replace=False, p=probs)
        for t in targets:
            edges.append((f"svc{i}", f"svc{t}", list(wl[t].ports)[0]))
            deg[t] += 1
    for i in range(0, n, 50):
        edges.append(("Internet", f"svc{i}", list(wl[i].ports)[0]))
    return Topology(wl, EXTERNALS, edges, ps)
