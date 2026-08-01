# Growth Lab operations

## Service contract

The production API is read only. It serves governed business metrics,
revenue forecasts, and MMM budget recommendations. It does not expose the
sealed simulator truth, arbitrary SQL, database mutation, model fitting, or
warehouse build commands.

| Route | Purpose | Authentication |
|---|---|---|
| `GET /healthz` | Process liveness and version | None |
| `GET /readyz` | Warehouse and MMM artifact readiness | None |
| `GET /metrics` | Prometheus text telemetry | API key |
| `POST /v1/metrics` | Governed semantic metrics | API key |
| `POST /v1/forecast` | Revenue forecast | API key |
| `POST /v1/budget` | MMM budget allocation | API key |

Production mode disables interactive API documentation and refuses to start
unless `GROWTH_LAB_API_KEY` contains at least thirty two characters.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GROWTH_LAB_ENV` | `development` | Runtime mode. Use `production` in deployments. |
| `GROWTH_LAB_API_KEY` | unset | Shared secret supplied through `X-API-Key`. |
| `GROWTH_LAB_DB` | `data/growth_lab.duckdb` | Read only warehouse path. |
| `GROWTH_LAB_MMM_PARAMS` | `models/mmm.json` | Validated MMM parameter artifact. |
| `GROWTH_LAB_MAX_REQUEST_BYTES` | `1048576` | Maximum declared request size. |
| `GROWTH_LAB_LOG_LEVEL` | `INFO` | JSON log severity threshold. |

Secrets belong in the deployment secret store. They must never be placed in
the image, Compose file, repository, or command history.

## Deployment

1. Build the image from a reviewed commit.

2. Record the commit SHA and image digest in the release record.

3. Inject a new API key from the platform secret store.

4. Run the container as user `10001` with a read only root filesystem. Drop
   all capabilities and enable the no new privileges control.

5. Wait for `/readyz` to return HTTP `200` before routing traffic.

6. Send a metrics query with a known channel filter and confirm that the
   response contains a request ID.

The included Compose profile demonstrates these controls for one host. A
managed orchestrator should reproduce the same user, filesystem, capability,
resource, secret, health, and readiness settings.

## Monitoring

Collect JSON logs from standard error. Preserve `request_id`, `route`,
`status`, and `elapsed_ms` fields. Scrape `/metrics` with the API key and alert
on sustained HTTP `500` responses, readiness failures, and latency changes.

Readiness requires both the DuckDB warehouse and MMM artifact. Liveness only
states that the process can respond. Do not restart a healthy process to fix a
readiness failure before checking mounted artifacts and permissions.

## Data and artifact lifecycle

The checked in simulator is deterministic, but the deployed DuckDB file and
MMM artifact are release assets. Build them once per release. Publish their
checksums beside the image digest. Keep the prior release assets until the new
release passes its smoke test.

The API opens DuckDB in read only mode for each request. Replace the database
through an atomic deployment or immutable image update. Never modify the live
file in place.

## Incident response

1. Remove traffic when readiness fails or error rates rise sharply.

2. Search logs by request ID and identify the affected route and release.

3. Rotate the API key immediately if disclosure is suspected.

4. Roll back the image and its paired data artifacts together.

5. Preserve logs, image digest, commit SHA, database checksum, and MMM artifact
   checksum for analysis.

6. Add a regression test before restoring the failed release path.

## Known boundary

API key authentication is appropriate for this portfolio service and simple
service to service deployment. A multi tenant deployment should place an
identity aware proxy in front of the API and authorize per principal. The
in process telemetry is per worker and resets on restart, so a real deployment
should scrape each worker and aggregate metrics externally.
