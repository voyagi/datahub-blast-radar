# Running DataHub locally for Blast Radar

Blast Radar needs a DataHub instance with lineage in it. This is the setup used
to develop it, including the two things that went wrong on Windows.

## 1. Start DataHub

```bash
uv tool install --python 3.11 acryl-datahub
datahub docker quickstart --version v1.6.0
```

Pin **v1.6.0** rather than taking the default. As of this writing the CLI's
default quickstart plan is v1.5.0.6, and two things need the newer server:

- the `showcase-ecommerce` sample pack sends aspects (`corpUserUsageFeatures`)
  that a v1.5 server rejects with `422 Unknown aspect`, and because proposals
  are sent in batches, one unknown aspect fails ~100 records with it;
- the MCP server's docs put DataHub Core v1.6.0+ as the floor.

Expect a multi-gigabyte pull on first run. When it finishes:

- UI: http://localhost:9002 (`datahub` / `datahub`)
- GMS: http://localhost:8080 - this is the URL Blast Radar wants, not the UI port

Quickstart runs with metadata service auth **disabled**, so a token is not
required locally. Confirm with:

```bash
curl -s -X POST http://localhost:8080/api/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"{ me { corpUser { username } } }"}'
```

A response naming `__datahub_system` means unauthenticated calls are accepted.

## 2. Load sample data

```bash
datahub datapack load showcase-ecommerce
```

That gives ~1,050 entities across Snowflake, Looker, PowerBI, and Tableau, with
the lineage, ownership, glossary terms, and domains the risk model reads.

### If you are on Windows

`datahub datapack load` and `datahub docker ingest-sample-data` both fail on
Windows with:

```
KeyError: 'Did not find a registered class for c'
```

The loader splits the pack's absolute path on `:` and reads the drive letter
(`c`) as an ingestion source type. Workaround: ingest the cached pack files
directly with a **relative** path, so no colon is involved.

The downloaded packs sit in `%USERPROFILE%\.datahub\datapack-cache\` as plain
JSON arrays of metadata change proposals. From inside that directory:

```yaml
# recipe.yml
source:
  type: file
  config:
    path: <cached-file>.json
sink:
  type: datahub-rest
  config:
    server: http://localhost:8080
```

```bash
datahub ingest -c recipe.yml
```

Load the structured-property definitions file before the main pack, so the
property values that reference them resolve.

## 3. Point Blast Radar at it

```bash
export DATAHUB_GMS_URL=http://localhost:8080
export TOOLS_IS_MUTATION_ENABLED=true
uv run blast-radar doctor
```

`doctor` connects through the MCP server and reports which read and write tools
the instance actually exposes.
