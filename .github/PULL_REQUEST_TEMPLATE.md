## What does this PR do?

<!-- One paragraph. Closes #123 -->

## Which rule(s) does it touch?

<!-- Hard rule numbers, or "none". -->

## How was it tested?

- [ ] `python -m psyconassist.cli demo --subject P-0001 --duration 300`
- [ ] `ruff check --select E9,F63,F7,F82 .`
- [ ] Added a fixture that fails without this change

## Checklist

- [ ] No real session data, transcripts, or API keys in the diff
- [ ] LLM still only called at `synth` / `risk`
- [ ] Severity logic stays in the rule table, not in prompts
