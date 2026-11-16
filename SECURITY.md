# Security boundary

FinGuard is intended for local synthetic-data research. Do not expose the Streamlit
server publicly: it has no authentication. API ingest and read keys must be different,
random secrets of at least 24 characters; authorization fails closed if unset.
Use environment variables; never commit keys or include them in screenshots.

Implemented controls: Pydantic input schemas, forbidden extra fields, finite/nonnegative
amount validation, unique IDs, strict event ordering, separate API roles, constant-time
key comparison, parameterized SQL, atomic transaction + ingestion audit writes,
model SHA-256 fingerprints, a 1,000-row request limit and 10,000-row demo ledger cap.
Only native XGBoost JSON is loaded; there is no untrusted pickle upload path.

Known limits: API keys have no per-user roles, rotation store or rate limiting; HTTP
is loopback-only by design. SQL data are unencrypted. Audit records can be altered by
anyone with filesystem access. No account-level tenancy or production secret manager
is implemented. The model can be evaded, the simulator may encode shortcuts, and
legitimate hubs can trigger gather/scatter alerts. Input records are syntactically
validated but their business authenticity is not established. Keys cannot protect a
compromised host. The request row limit does not replace an HTTP body-size limit.

Before any real deployment: independent validation on appropriately governed data,
OIDC/RBAC, encrypted storage and TLS, per-tenant isolation, immutable external audit
storage, request/rate limits, late-event processing, feature-store consistency,
model monitoring, calibration, human review and incident response are required.
This repository deliberately performs no consequential financial actions.
