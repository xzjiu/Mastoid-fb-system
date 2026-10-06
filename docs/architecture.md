# Architecture

## System boundary

The system converts a recorded simulator session into inspectable evidence and
post-session feedback. Each layer has one responsibility:

1. **Source adapters** read a configured run, anatomy resources, and optional
   human-authored phase boundaries without modifying the source files.
2. **Frame pipeline** derives normalized frame observations, geometry context,
   visibility observations, and motion segments.
3. **Event and relation builders** group observations and create factual links
   between events, phases, anatomy, motion, and source evidence.
4. **Evidence store and retrieval** persist canonical artifacts and expose only
   allowlisted, read-only queries.
5. **Feedback and validation** organize retrieved evidence, preserve candidate
   status, require citations, and block unsupported causal or clinical claims.
6. **Local web applications** present feedback and linked media without making
   source data part of either frontend bundle.

```text
configured sources
       |
       v
frame evidence -> events -> relations -> evidence store
                                      |          |
                                      +---- retrieval
                                                |
                                                v
                                      validated feedback -> local web app
```

## Official applications

The repository contains two official static applications over the same local
service and validated evidence contract:

- `apps/expert_study/` provides the complete expert-study journey and stores
  its study inputs separately from canonical evidence.
- `apps/recommendations/` presents recommendation-only feedback and does not
  expose expert annotation or item-level review controls.

Keeping these interfaces separate prevents study data-collection behavior from
silently entering the recommendation-only experience.

## Configuration

Repository configuration contains only shareable policies and synthetic
examples. Dataset manifests, anatomy mappings, source roots, and case-specific
review decisions are injected at runtime from ignored local files. Production
code must not assume a particular dataset, case identifier, directory layout,
or operating system.

## Determinism and external models

Evidence construction, storage, retrieval, and safety checks are deterministic.
An external text model may select an allowlisted retrieval intent or draft text
from a deidentified evidence packet, but it cannot create evidence, change an
evidence status, or bypass claim validation. The deterministic path must remain
fully usable without network access or model credentials.

## Portability

Paths are represented with `pathlib` and supplied through configuration. Python
3.12 is tested on Windows, macOS, and Linux. Generated artifacts live outside
the repository, and tests build synthetic inputs at runtime so that CI does not
depend on private files.
