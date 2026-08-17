# kb1 distiller

You are adding paper notes to the kb1 corpus at `{kb1}` from the sources at
`{sources}`.

{instruction}

kb1 is what block agents read before choosing a line of attack. A note is
not a summary for a human — it is a working reference for an agent that has
480 seconds of GPU time and needs to know whether this method is worth
trying and how to implement it in JAX. Write for that reader.

## Node format

One file per paper, named `NNN_YYYY_Title_words_joined_by_underscores.md`,
where `NNN` is the next free three-digit id (chronological order: read
`{kb1}/INDEX.md` for the highest in use) and the title is truncated to keep
the filename around 60 characters.

Frontmatter, then exactly four sections — no more, no fewer:

    ---
    slot: <integer id, matching NNN>
    title: "<full paper title>"
    authors: [<Family Name>, <Family Name>]
    year: <YYYY>
    venue: "<journal or conference (arXiv:NNNN.NNNNN)>"
    gitrepo: "<url, or empty string>"
    ---

    ## TL;DR
    ## Problem
    ## Method
    ## Results

## Length

Match the corpus. The 130 existing nodes run **3.6–6.3 KB**, median **5.0 KB**
— roughly 700 words, of which the Method section is usually half. Aim for
that band and treat **4–6 KB** as the target.

This is not tidiness. A block reads several nodes before it picks a line of
attack; a node twice the size of its neighbours costs the reader context it
needs for the work itself, and one half the size has almost always dropped
the runnable Method that made it worth opening. `pinnforge kb1 check` flags
anything outside 3–8 KB.

## What goes in each section

**TL;DR** — one paragraph. The mechanism and the claim, concretely enough
that a reader can decide from this alone whether to open the file. Name the
actual quantity that changes, not "an improved approach".

**Problem** — what fails without this method, stated in terms of PINN
training behaviour: spectral bias, stiff gradients, an ill-conditioned loss,
a boundary condition that soft penalties cannot enforce. If the paper's
motivation does not translate into something a PINN run would exhibit, say
what it does address.

**Method** — the heart of the note, and the reason a block opens the file.
Give the equations in LaTeX (`$$ … $$`), define every symbol, state the
initialisation and any hyperparameters with the values the paper used. Then
give **runnable JAX** — `flax.linen` + `optax`, matching the stack the tasks
are built on — showing the part that differs from a vanilla PINN. Not
pseudocode: an agent will copy it.

**Results** — what was measured, on which equations, against what baseline,
with the numbers. Include the cost: if the method doubles step time, that
decides whether it fits a wall-clock budget. Record the failure cases and
limits the paper reports; a note that only carries the good news will send
blocks down dead ends.

## Rules

- One paper per file. Do not merge related papers into one node.
- Cite nothing you have not read in the source material. If the source is a
  paper you only have an abstract for, say so in TL;DR rather than inventing
  a Method section.
- Keep the corpus's voice: factual, dense, no hedging, no marketing.
- Never edit an existing node's numbering. Ids are stable references — block
  summaries cite them.

## INDEX.md

After writing the nodes, append one entry per new paper to `{kb1}/INDEX.md`,
in id order, matching the existing shape exactly:

    - **NNN_YYYY_Title.md** (YYYY) — <full title>
      <TL;DR truncated to ~250 characters, ending in …>

Update the paper count in the file's first line.

## Report

Finish by listing the ids and titles you added, and note any paper you
skipped and why.
