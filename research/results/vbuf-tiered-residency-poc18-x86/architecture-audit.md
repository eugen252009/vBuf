# POC18 Architecture Audit

`ResidentTensorMaterializer` owns request/lease bookkeeping and delegates new
payload acquisition to the backing `TensorMaterializer`. `TensorResidencyStore`
owns retained payload handles, identity keys, byte accounting, recency, and
eviction. A lease protects execution validity; it does not change persistent
identity.

POC18 adds `TieredResidencyStore` as a separate generic contract. It owns tier
placement and generation metadata only for the host-emulated qualification
seam. The model `PersistentTensorRef` identity remains the same integer key
through `BACKING -> WARM -> HOT -> WARM -> BACKING`.

The tier contract is exclusive by default: one identity occupies at most one
resident tier, so duplicate resident payload bytes are zero. Promotion and
demotion are explicit, immutable-weight transfers with separately accounted
bytes. Active leases prevent movement and drop. Runtime state is not accepted
by this contract.

The existing CPU materializer remains unchanged in execution semantics. The
POC18 real-workload result is an exact replay of the POC17 logical request
stream, with POC16 execution/parity and failure evidence retained as the
semantic control.
