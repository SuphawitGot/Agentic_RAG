# CLAUDE.md

This project is worked on from two PCs, each with its own Claude session.
`ROADMAP.md` is the shared record of where the work stands. Keep it current.

## Start of a session

1. Run `git pull` so you have the other PC's latest work.
2. Read the **Current status** and **Progress log** sections of `ROADMAP.md`
   before doing anything else, and pick up from there.

## While working

After each finished piece of work (a step done, a bug fixed, a decision made,
or the session stopping partway), update `ROADMAP.md`:

- **Current status:** current phase, what is in progress, the next step, and
  any blockers.
- **Progress log:** add a new entry at the top with the date, which PC
  (hostname/OS), what changed, and what is left.
- Tick off or adjust the phase bullets when a step is done.

Keep entries short. Record decisions and their reasons under **Open
decisions** or in the log entry.

## Syncing

The other PC only sees the update once it is pushed. Commit `ROADMAP.md`
together with the related code change and push to `main`, after asking the
user if they haven't already approved pushing.

## Commands

- Install: `uv sync`
- Tests: `uv run python -m unittest discover -s tests`
- Agent: `uv run python -m ragsale.agent "..."`
