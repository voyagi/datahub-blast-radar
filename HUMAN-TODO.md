# Human to-do

Everything here needs a person: an account, a camera, a running DataHub, or a
decision that is not mine to make. Everything that did not is already done.

Deadline for the first three: **2026-08-10, 23:00 CEST**.

## Before the submission

- [ ] **Record the demo video.** Script, shot list and timings are in
      `docs/submission.md`. `scripts/demo-run.ps1` drives the three terminal
      steps so nothing has to be typed on camera. Under 3:00, upload to
      YouTube, paste the link into Devpost and into the "Try it out" section
      of `docs/submission.md`.
- [ ] **Submit on Devpost.** The copy is written and current; it needs an
      account, the gallery images from `docs/screenshots/` (lead with
      `datahub-tag-on-asset.png`), and the video link.
- [ ] **Run `blast-radar setup` and one `scan --publish` against a live
      DataHub.** Two things in this branch have only ever run against tests:
      the GraphQL POST was rewritten onto `http.client`, and
      `SCORED_ENTITY_TYPES` gained `urn:li:entityType:datahub.mlFeatureTable`.
      Both are unit-tested, and neither has been answered by a real GMS. If
      the property definition is rejected for that entity type, `setup` says
      so in one line rather than failing.
- [ ] **Check the MCP server log for the token.** The server logs every
      GraphQL query it sends to `.blast-radar/mcp-server.log` at DEBUG. That
      directory is gitignored, but whether the server writes the
      `Authorization` header into it is its behavior, not ours, and it can
      only be answered by looking at a real one.

## Before this repo is treated as clean

- [ ] **The git history still carries a personal email.** All 21 commits on
      `main` before this branch were authored under a personal address on a
      public repo (see `git log --format='%ae' main` for the exact value). The
      repo-local identity is fixed, so every commit from this branch onward is
      clean, but history is not rewritten by that. The agreed fix is to squash
      to one commit and delete and recreate the repository at the same URL,
      after this branch merges. It is irreversible, it drops the 21-commit
      history, and it needs saying out loud before it runs.

## Optional, not blocking

- [ ] Column-level lineage via `get_lineage_paths_between`, so the score
      answers "this asset reads the column being dropped" rather than "this
      asset is downstream of the change". This is the honest limit of the
      current model and it is written down in the README.
- [ ] A severity tag currently reflects the last scan that flagged the asset.
      An asset that drops out of the flagged band keeps its tag until
      something flags it again, because nothing records which assets a given
      scan tagged.
