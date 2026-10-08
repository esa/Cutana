[//]: # (Copyright © European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Performance and memory

Cutana is designed to use the available memory and CPU cores fully without overloading the system. A load balancer monitors the system while a run is going and adjusts the number of worker processes and their memory allocation.

## Memory is usually the limit

In most deployments, **memory runs out before CPU cores do**. The load balancer continuously monitors memory use and changes the number of worker processes to avoid exhausting it. If memory pressure increases, it scales processing down.

The default `max_workers` is the number of CPUs your process may actually use, which honours Kubernetes and cgroup limits. On ESA Datalabs that is the pod's CPU limit, not the node's core count.

!!! warning "Don't run other memory-intensive work alongside Cutana"
    Competing memory-intensive processes interfere with the load balancer and can lead to:

    - crashes from memory exhaustion
    - slower processing
    - inconsistent or failed results
    - in extreme cases, corrupted data

## Getting the best performance

1. **Dedicated resources**: run Cutana on dedicated compute resources when possible.
2. **Monitoring**: watch memory use with system monitoring tools during long runs.
3. **Batch size**: adjust `N_batch_cutout_process` to the memory available.
4. **Worker count**: let the load balancer choose, or set `max_workers` conservatively.

Preventing memory pressure through resource management is better than relying on the load balancer to react to it.

## Choosing a backend

| Backend | When | Trade-off |
| --- | --- | --- |
| `create_cutouts_direct()` | Fewer than about 1,000 sources | Fastest to start; holds every result in memory |
| `Orchestrator` | Full catalogues written to disk | Lightest on memory |
| `StreamingOrchestrator` | Survey-scale catalogues and ML pipelines | Highest throughput at scale; memory stays bounded |

How much faster streaming is depends on the cost of each cutout. Small single-band cutouts are dominated by per-process overhead, which the streaming backends keep low. Large multi-band cutouts are dominated by reading and resizing, which every backend pays.
