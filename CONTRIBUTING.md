# Contributing to PsyConAssist

This project touches mental-health data, so the bar is higher than usual.

## Ground rules

- **Never commit session data.** No videos, transcripts, audio, profiles, or screenshots of real sessions.
- **No clever prompting in place of rules.** If a behaviour matters, it belongs in the rule table, not in a prompt.
- **The LLM stays behind its two doors** (`synth`, `risk`). PRs that call a model elsewhere will be asked to change.

## Development setup

```bash
python -m venv .venv && . .venv/Scripts/activate    # Windows
pip install -r requirements.txt
python -m psyconassist.cli demo --subject P-0001 --duration 600
```

Optional engines (only needed for real video):

```bash
pip install -e ".[media]"
# plus ffmpeg on PATH, and OpenFace configured via OPENFACE_BIN
```

## What makes a good PR here

1. **A failing fixture first.** Add a synthetic session (or a hand-built chunk) that violates a rule, show it slipping through, then fix it.
2. **One rule per PR.** Don't bundle three rule changes with a UI change.
3. **Evidence in the description.** Show the event JSON before / after.

## Checks before opening a PR

```bash
ruff check --select E9,F63,F7,F82 .
python -m compileall -q psyconassist
python -m psyconassist.cli demo --subject P-0001 --duration 300
```

## Commit messages

```
s3: treat long silences after a risk event as high severity
ui: keep timeline scroll position after feedback submit
```

## License

Contributions are licensed under the [MIT License](LICENSE).
