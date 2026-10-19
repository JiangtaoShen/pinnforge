# The discovery film of run {run}

Run `{run}` is over. Its record will be played as a short film by a fixed
player: the same panels, layout, colours and timing for every run. You are
writing the film's **script**: the words it shows. The player takes every
number (scores, times, costs, which candidate beat which) from the run's
records. You never write numbers for it.

Work only in this directory, `{pkg}`. Do not read anything outside it, and
do not run training or evaluations. Everything you need is here.

## What is here

| file | what it is |
|---|---|
| `facts.json` | The run as the film will show it, built from its records. This is the source of truth. `blocks[]` gives each block's candidates (`roster`: id, score `r`, status `st`), the champion score before (`entry`) and after (`exit`), costs and counts. `kb1.entries` lists the theory corpus (paper `id`, `y` year, `t` title). |
| `draft.json` | A first script, extracted mechanically from the block summaries. It already has the right shape and ids. Improve it; do not just hand it back. |
| `summaries/bNN.md` | What each block wrote about itself: thrust, the table of its evaluations, findings. |
| `activity/bNN.md` | What each block's agent did, step by step: its reasoning, the files it read and wrote, the commands it ran. |
| `problem.md` | The task. `kb1/INDEX.md` is the corpus the blocks could read. |
| `example/script.json` | A finished script from another run, on another task. It is the paradigm: follow its shape, length and tone. Its ids, papers and numbers belong to that run, so never copy them. |
| `check.py` | The check your script must pass: `{python} check.py`. |

## What the film shows, and where your words go

The film plays the run block by block. For each block:

- **Agent session panel.** The `headline` is shown in bold, the `thrust` is
  typed out under it, and the `tags` appear as chips. The first `findings`
  item is shown as a quote.
- **Candidate roster.** Each candidate is listed with its score. Your
  `candidates` label (for example `CP rank 384 → 768`) replaces the bare
  file id, so say what the candidate changed, not its name.
- **Knowledge base 1 (literature).** The papers in `cited` light up as the
  block reaches for them.
- **Knowledge base 2 (practice).** The `findings` are written in as the notes
  this block leaves for later blocks.

At the end a closing card shows the result and your `logline`: the whole run
in one sentence.

## The script: `script.json`

```json
{
  "version": 1,
  "logline": "the run in one sentence (≤ {max_logline} characters)",
  "blocks": {
    "b00": {
      "headline": "≤ {max_headline} characters",
      "thrust": "≤ {max_thrust} characters",
      "tags": ["≤ {max_tags} tags, each ≤ {max_tag} characters"],
      "findings": ["≤ {max_findings} items, each ≤ {max_finding} characters"],
      "cited": ["paper ids from facts.json kb1.entries, ≤ {max_cited}"],
      "candidates": {"candidate id from this block's roster": "label ≤ {max_label} characters"}
    }
  }
}
```

Every block in `facts.json` needs an entry: {blocks}.

## How to write it

1. Read `facts.json` first, then each block's summary and activity, in
   order. Follow the line of the run: what each block inherited, what it
   tried, what moved the score, what did not, and why.
2. **Headline:** `bNN — what the block did: entry → exit`, with the two
   champion scores from `facts.json` when it moved the champion. Name the
   idea, not the file.
3. **Thrust:** its line of attack in two or three sentences. Say where it
   forked from, what it set out to test, and why that was the next thing to
   try.
4. **Findings:** what a later block, or a reader, should take away. Write
   transferable lessons, not a log. Keep the strongest one first, because it
   is the one quoted on screen.
5. **Cited:** only papers the block demonstrably used, shown by its summary
   or its activity (a corpus file it opened, a paper it named). Leave the
   list empty rather than guess.
6. **Numbers in your text** must come from `facts.json` or the summaries,
   rounded the way they are shown there. Never estimate one.
7. Write plain English text: no Markdown, no HTML, no emoji. Be concise,
   specific and technical, as in the example.
8. Run `{python} check.py` and fix everything it reports. You are done when it
   prints `ok: the script is ready`. Then stop.
